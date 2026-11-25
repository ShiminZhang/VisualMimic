"""Check video metadata and produce contact sheets before visual assessment."""
import json
from pathlib import Path
import hashlib
import cv2
import numpy as np
from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parent
ROWS = json.loads((OUT / 'inputs.json').read_text())
NAMES = ['SOURCE', 'A_human', 'B_human', 'A_reference', 'B_reference', 'A_execution', 'B_execution']
records = []
(OUT / 'qa').mkdir(exist_ok=True)
for start in range(0, len(ROWS), 5):
    canvas = Image.new('RGB', (1008, 700), 'white')
    draw = ImageDraw.Draw(canvas)
    for c, name in enumerate(NAMES): draw.text((c * 144 + 3, 5), name, fill='black')
    for row_index, row in enumerate(ROWS[start:start + 5]):
        sid = row['key']['sample_id']
        y = 25 + row_index * 135
        draw.text((3, y), sid, fill='black')
        videos = {'SOURCE': row['source_video']}
        for name in NAMES[1:]: videos[name] = json.loads((OUT / 'videos' / sid / (name + '.json')).read_text())
        assert videos['A_human']['camera_position_yup'] == videos['B_human']['camera_position_yup']
        assert videos['A_human']['camera_target_yup'] == videos['B_human']['camera_target_yup']
        assert all(videos[name]['camera'] == videos['A_reference']['camera'] for name in NAMES[3:])
        for column, name in enumerate(NAMES):
            video = videos[name]
            p = Path(video['video']['uri'])
            assert hashlib.sha256(p.read_bytes()).hexdigest() == video['video']['sha256']
            cap = cv2.VideoCapture(str(p))
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            assert count == len(video['timestamps_s'])
            frame = None
            for i in sorted({0, count // 2, count - 1}):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, sample = cap.read()
                assert ok and sample.shape == (480, 640, 3)
                assert np.std(sample) > 3, (sid, name, 'blank frame')
                if i == count // 2: frame = sample
            cap.release()
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((144, 108))
            canvas.paste(image, (column * 144, y + 17))
            records.append({'sample_id': sid, 'condition': name, 'frames': count, 'duration_s': video['timestamps_s'][-1] - video['timestamps_s'][0], 'sha256': video['video']['sha256']})
    canvas.save(OUT / 'qa' / ('overview-' + str(start // 5 + 1) + '.png'))
(OUT / 'qa' / 'video_checks.json').write_text(json.dumps({'videos': len(records), 'cases': len(ROWS), 'matched_cameras': True, 'records': records}, indent=2))
print('VERIFIED', len(records), 'videos; contact sheets ready', flush=True)
