#!/usr/bin/env python3
"""Save debug PNGs for specified annotation frames with overlays.

Usage: python scripts/save_debug_frames.py
"""
import os
from collections import defaultdict
import csv

import imageio
from PIL import Image, ImageDraw, ImageFont

import csv

# small helpers copied from scripts/generate_overlay.py to avoid import issues
LABEL_RADIUS_MAP = {
    'pedestrian': 10,
    'person': 10,
    'biker': 14,
    'cyclist': 14,
    'car': 26,
    'bus': 36,
    'truck': 36,
}


def get_label_radius(label):
    if not label:
        return 12
    key = str(label).strip().lower()
    return LABEL_RADIUS_MAP.get(key, 12)


def load_annotations(annot_path):
    ann_map = {}
    if not os.path.exists(annot_path):
        raise FileNotFoundError(annot_path)
    with open(annot_path, newline='') as f:
        rdr = csv.DictReader(f)
        cols = rdr.fieldnames
        for row in rdr:
            try:
                fr = int(row.get('frame') or row.get('Frame') or 0)
            except Exception:
                fr = 0
            try:
                tid = int(row.get('track_id') or row.get('track') or row.get('id') or 0)
            except Exception:
                tid = 0
            if 'xmin' in cols and 'xmax' in cols and 'ymin' in cols and 'ymax' in cols:
                left = float(row.get('xmin', 0))
                top = float(row.get('ymin', 0))
                w = float(row.get('xmax', 0)) - float(row.get('xmin', 0))
                h = float(row.get('ymax', 0)) - float(row.get('ymin', 0))
            else:
                left = float(row.get('left') or row.get('x') or 0)
                top = float(row.get('top') or row.get('y') or 0)
                w = float(row.get('width') or row.get('w') or 0)
                h = float(row.get('height') or row.get('h') or 0)
            label = row.get('label','')
            ann_map[(fr, tid)] = (left, top, w, h, label)
    return ann_map


OUT_DIR = 'outputs'
FRAMES = [4000, 4265, 4530]
VIDEO = 'SDD_datasets/SDD_video/coupa/video0/video.mov'
ANN = 'SDD_datasets/SDD_raw/coupa/video0/annotations.csv'
SCORED = 'outputs/pedestrian_1_scored.csv'


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # load scored csv (fallback to csv reader to avoid pandas dependency)
    scored_rows = []
    with open(SCORED, newline='') as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            scored_rows.append(r)


    # load ann map using local loader
    ann_map = load_annotations(ANN)
    min_ann = min((k[0] for k in ann_map.keys())) if ann_map else 0

    reader = imageio.get_reader(VIDEO, 'ffmpeg')
    meta = reader.get_meta_data()
    n_vid = meta.get('nframes', None)

    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 14)
    except Exception:
        font = ImageFont.load_default()

    def draw_text_with_bg(draw, xy, text):
        tb = draw.textbbox((0,0), text, font=font)
        w = tb[2]-tb[0]
        h = tb[3]-tb[1]
        x,y = xy
        pad = 3
        draw.rectangle((x-pad,y-pad,x+w+pad,y+h+pad), fill=(255,255,255,200))
        draw.text((x,y), text, font=font, fill=(0,0,0))

    for fr in FRAMES:
        # compute video index with same mapping as generate_overlay
        if (n_vid is not None) and (0 <= fr < n_vid):
            vid_idx = int(fr)
        else:
            alt = fr - min_ann
            if (n_vid is not None) and (0 <= alt < n_vid):
                vid_idx = int(alt)
            else:
                vid_idx = max(0, min(int(fr), n_vid-1)) if n_vid is not None else int(fr)

        frame = reader.get_data(vid_idx)
        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil, 'RGBA')

        # draw target first (use same logic as generate_overlay)
        # determine target id (from scored_rows)
        target_id = None
        try:
            if len(scored_rows) > 0:
                # find first non-empty target_track_id
                for r in scored_rows:
                    if r.get('target_track_id'):
                        target_id = int(r.get('target_track_id'))
                        break
        except Exception:
            target_id = None

        if target_id is not None:
            tbbox = ann_map.get((fr, int(target_id)))
            if tbbox is not None:
                if len(tbbox) == 5:
                    t_left, t_top, t_w, t_h, t_label = tbbox
                else:
                    t_left, t_top, t_w, t_h = tbbox
                    t_label = ''
                side = max(t_w, t_h)
                cx_t = t_left + t_w/2.0
                cy_t = t_top + t_h/2.0
                sq_left = cx_t - side/2.0
                sq_top = cy_t - side/2.0
                sq_right = cx_t + side/2.0
                sq_bottom = cy_t + side/2.0
                draw.rectangle((sq_left, sq_top, sq_right, sq_bottom), fill=(0,0,0,255), outline=(255,255,255,255))
                id_text = str(target_id)
                tb = draw.textbbox((0,0), id_text, font=font)
                tw = tb[2]-tb[0]; th = tb[3]-tb[1]
                draw.text((cx_t-tw/2, cy_t-th/2), id_text, font=font, fill=(255,255,255))

            else:
                # fallback: look up target center from scored_rows for this frame
                tmp = [r for r in scored_rows if int(r.get('frame') or -9999) == fr]
                if tmp:
                    try:
                        cx_t = float(tmp[0].get('target_cx') or 0)
                        cy_t = float(tmp[0].get('target_cy') or 0)
                        side = 40
                        sq_left = cx_t - side/2.0
                        sq_top = cy_t - side/2.0
                        sq_right = cx_t + side/2.0
                        sq_bottom = cy_t + side/2.0
                        draw.rectangle((sq_left, sq_top, sq_right, sq_bottom), fill=(0,0,0,255), outline=(255,255,255,255))
                        id_text = str(target_id)
                        tb = draw.textbbox((0,0), id_text, font=font)
                        tw = tb[2]-tb[0]; th = tb[3]-tb[1]
                        draw.text((cx_t-tw/2, cy_t-th/2), id_text, font=font, fill=(255,255,255))
                    except Exception:
                        pass

        # draw others from scored CSV for this frame
        rows_here = [r for r in scored_rows if int(r.get('frame') or -9999) == fr]
        # compute smin/smax from scored_rows
        scores = [float(r.get('score') or 0.0) for r in scored_rows if r.get('score') not in (None, '')]
        smin = min(scores) if scores else 0.0
        smax = max(scores) if scores else 1.0

        try:
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            cmap = cm.get_cmap('Reds')
            norm = mcolors.Normalize(vmin=smin, vmax=smax)
        except Exception:
            cmap = None
            norm = None

        min_alpha = 80
        max_alpha = 230

        for row in rows_here:
            other_id = int(row.get('other_track_id') or 0)
            score = float(row.get('score') or 0.0)
            bbox = ann_map.get((fr, other_id)) or ann_map.get((fr, int(other_id)))
            if bbox is None:
                continue
            if len(bbox) == 5:
                left, top, w, h, ann_label = bbox
            else:
                left, top, w, h = bbox
                ann_label = ''
            cx = left + w/2.0
            cy = top + h/2.0

            label = row.get('other_label') or ann_label or ''
            radius = get_label_radius(label)

            if smax > smin:
                frac = (score - smin) / (smax - smin)
            else:
                frac = 0.5
            alpha = int(min_alpha + frac * (max_alpha - min_alpha))
            if cmap is not None and norm is not None:
                rgba = cmap(norm(score))
                col = tuple(int(255*c) for c in rgba[:3]) + (alpha,)
            else:
                col = (255,0,0,alpha)

            draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=col, outline=(0,0,0,180))
            id_text = str(other_id)
            tb = draw.textbbox((0,0), id_text, font=font)
            tw = tb[2]-tb[0]; th = tb[3]-tb[1]
            draw.text((cx-tw/2, cy-th/2), id_text, font=font, fill=(0,0,0))
            score_text = f"{score:.2f}"
            draw_text_with_bg(draw, (cx+radius+6, cy-th/2), score_text)

        outp = os.path.join(OUT_DIR, f'debug_frame_{fr}.png')
        pil.save(outp)
        print('Saved', outp)

    reader.close()


if __name__ == '__main__':
    main()
