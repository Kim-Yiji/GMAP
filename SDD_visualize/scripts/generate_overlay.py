#!/usr/bin/env python3
"""Generate overlay MOV showing scored co-appearances for a target pedestrian.

Usage:
  python scripts/generate_overlay.py \
      --video SDD_datasets/SDD_video/coupa/video0/video.mov \
      --ann SDD_datasets/SDD_raw/coupa/video0/annotations.csv \
      --scored outputs/pedestrian_1_scored.csv \
      --out outputs/pedestrian_1_overlay_on_video_fixed.mov

This script uses imageio (ffmpeg backend) and Pillow to draw overlays.
It prefers direct mapping annotation_frame -> video frame index when possible,
and falls back to (frame - min_annotation_frame) when necessary.
"""
import argparse
import os
from collections import defaultdict
import csv

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors
except Exception:
    cm = None
    mcolors = None

import imageio

# fixed radius per label (in pixels) — adjust as desired
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
    if pd is not None:
        ann_df = pd.read_csv(annot_path)
        cols = list(ann_df.columns)
        # common SDD format: xmin,ymin,xmax,ymax,frame,track_id,label
        if all(c in ann_df.columns for c in ("xmin","xmax","ymin","ymax","frame","track_id")):
            for _, r in ann_df.iterrows():
                fr = int(r['frame'])
                tid = int(r['track_id'])
                left = float(r['xmin'])
                top = float(r['ymin'])
                w = float(r['xmax']) - float(r['xmin'])
                h = float(r['ymax']) - float(r['ymin'])
                label = r['label'] if 'label' in ann_df.columns else ''
                ann_map[(fr, tid)] = (left, top, w, h, label)
        else:
            # try alternative names (left,top,width,height)
            def pick(names):
                for n in names:
                    if n in ann_df.columns:
                        return n
                return None
            fcol = pick(['frame','Frame'])
            tcol = pick(['track_id','track','id'])
            lcol = pick(['left','x','xmin'])
            tcolp = pick(['top','y','ymin'])
            wcol = pick(['width','w'])
            hcol = pick(['height','h'])
            if None in (fcol,tcol,lcol,tcolp,wcol,hcol):
                raise RuntimeError(f"Unsupported annotation columns: {cols}")
            for _, r in ann_df.iterrows():
                fr = int(r[fcol])
                tid = int(r[tcol])
                left = float(r[lcol])
                top = float(r[tcolp])
                w = float(r[wcol])
                h = float(r[hcol])
                label = r.get('label','') if 'label' in ann_df.columns else ''
                ann_map[(fr, tid)] = (left, top, w, h, label)
    else:
        # fallback CSV reader
        with open(annot_path, newline='') as f:
            rdr = csv.DictReader(f)
            cols = rdr.fieldnames
            for row in rdr:
                fr = int(row.get('frame') or row.get('Frame') or 0)
                tid = int(row.get('track_id') or row.get('track') or 0)
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


def draw_overlay(video_path, annot_path, scored_csv, out_path, target_track_id=None, max_frames=None):
    if pd is not None:
        df_scores = pd.read_csv(scored_csv)
    else:
        # minimal fallback
        rows = []
        with open(scored_csv, newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
        df_scores = None

    ann_map = load_annotations(annot_path)
    if ann_map:
        min_ann = min(k[0] for k in ann_map.keys())
    else:
        min_ann = 0

    # frames to process
    if df_scores is not None:
        frames = sorted(df_scores['frame'].unique().tolist())
        smin = float(df_scores['score'].min())
        smax = float(df_scores['score'].max())
    else:
        frames = sorted({int(r['frame']) for r in rows})
        smin = min(float(r.get('score',0)) for r in rows)
        smax = max(float(r.get('score',0)) for r in rows)

    # optional truncation for quick previews
    if max_frames is not None:
        frames = frames[:int(max_frames)]

    # colormap
    if cm is not None and mcolors is not None:
        cmap = cm.get_cmap('Reds')
        norm = mcolors.Normalize(vmin=smin, vmax=smax)
    else:
        cmap = None
        norm = None

    # opacity mapping for circles: score -> alpha (0-255)
    min_alpha = 80
    max_alpha = 230

    # target drawing parameters
    # TARGET_SCALE: multiply bbox max(w,h) by this to get square side (<=1.0 makes it smaller)
    TARGET_SCALE = 0.7
    # TARGET_ALPHA: 0..255 (lower -> more transparent)
    TARGET_ALPHA = 180
    # fallback square side when no bbox is available
    TARGET_FALLBACK_SIDE = 30

    reader = imageio.get_reader(video_path, 'ffmpeg')
    meta = reader.get_meta_data()
    fps = meta.get('fps', 30)
    n_vid = meta.get('nframes', None)
    print(f'Video fps={fps}, nframes={n_vid}, frames_to_process={len(frames)}')

    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', ffmpeg_params=['-pix_fmt','yuv420p'])

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

    processed = 0
    for fr in frames:
        if (n_vid is not None) and (0 <= fr < n_vid):
            vid_idx = int(fr)
        else:
            alt = fr - min_ann
            if (n_vid is not None) and (0 <= alt < n_vid):
                vid_idx = int(alt)
            else:
                vid_idx = max(0, min(int(fr), n_vid-1)) if n_vid is not None else int(fr)

        try:
            frame = reader.get_data(vid_idx)
        except Exception as e:
            print(f'Could not read video frame {vid_idx} for annotation frame {fr}: {e}')
            continue

        pil = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil, 'RGBA')
        # determine target track id if not explicitly provided
        if target_track_id is None and df_scores is not None:
            try:
                unique_targets = df_scores['target_track_id'].unique()
                if len(unique_targets) > 0:
                    target_track_id = int(unique_targets[0])
            except Exception:
                pass

        # draw target (prefer smaller, semi-transparent square so underlying video remains visible)
        if target_track_id is not None:
            tbbox = ann_map.get((fr, int(target_track_id)))
            if tbbox is not None:
                if len(tbbox) == 5:
                    t_left, t_top, t_w, t_h, t_label = tbbox
                else:
                    t_left, t_top, t_w, t_h = tbbox
                    t_label = ''
                # draw a black circle for the target (smaller, semi-transparent)
                side = max(t_w, t_h) * TARGET_SCALE
                if side <= 0:
                    side = TARGET_FALLBACK_SIDE
                cx_t = t_left + t_w/2.0
                cy_t = t_top + t_h/2.0
                r = side / 2.0
                draw.ellipse((cx_t-r, cy_t-r, cx_t+r, cy_t+r), fill=(0,0,0,TARGET_ALPHA), outline=(255,255,255,200))
                id_text = str(target_track_id)
                tb = draw.textbbox((0,0), id_text, font=font)
                tw = tb[2]-tb[0]; th = tb[3]-tb[1]
                draw.text((cx_t-tw/2, cy_t-th/2), id_text, font=font, fill=(255,255,255))
            else:
                # fallback: try to get center from scored CSV rows for this frame
                if df_scores is not None:
                    tmp = df_scores[df_scores['frame']==fr]
                    if not tmp.empty:
                        try:
                            cx_t = float(tmp.iloc[0]['target_cx'])
                            cy_t = float(tmp.iloc[0]['target_cy'])
                            side = 40
                            r = side/2.0
                            draw.ellipse((cx_t-r, cy_t-r, cx_t+r, cy_t+r), fill=(0,0,0,TARGET_ALPHA), outline=(255,255,255,200))
                            id_text = str(target_track_id)
                            tb = draw.textbbox((0,0), id_text, font=font)
                            tw = tb[2]-tb[0]; th = tb[3]-tb[1]
                            draw.text((cx_t-tw/2, cy_t-th/2), id_text, font=font, fill=(255,255,255))
                        except Exception:
                            pass

        if df_scores is not None:
            rows_here = df_scores[df_scores['frame']==fr].to_dict('records')
        else:
            rows_here = [r for r in rows if int(r['frame'])==fr]

        for row in rows_here:
            other_id = int(row.get('other_track_id')) if df_scores is not None else int(row.get('other_track_id') or 0)
            target_id = int(row.get('target_track_id')) if df_scores is not None else int(row.get('target_track_id') or 0)
            score = float(row.get('score') or 0.0)

            bbox = ann_map.get((fr, other_id)) or ann_map.get((fr, int(other_id)))
            if bbox is None:
                continue
            # ann_map stores (left, top, w, h, label)
            if len(bbox) == 5:
                left, top, w, h, ann_label = bbox
            else:
                left, top, w, h = bbox
                ann_label = ''
            cx = left + w/2.0
            cy = top + h/2.0

            if other_id == target_id:
                # make the target square smaller and slightly transparent so underlying video may be visible
                side = max(w, h) * TARGET_SCALE
                if side <= 0:
                    side = TARGET_FALLBACK_SIDE
                sq_left = cx - side/2.0
                sq_top = cy - side/2.0
                sq_right = cx + side/2.0
                sq_bottom = cy + side/2.0
                draw.rectangle((sq_left, sq_top, sq_right, sq_bottom), fill=(0,0,0,TARGET_ALPHA), outline=(255,255,255,200))
                id_text = str(target_id)
                tb = draw.textbbox((0,0), id_text, font=font)
                tw = tb[2]-tb[0]; th = tb[3]-tb[1]
                draw.text((cx-tw/2, cy-th/2), id_text, font=font, fill=(255,255,255))
            else:
                # determine label (prefer scored CSV label if present)
                label = ''
                if df_scores is not None:
                    label = row.get('other_label') or ann_label or ''
                else:
                    label = row.get('other_label') or ann_label or ''

                # fixed radius based on label (ignore bbox size for consistent visuals)
                radius = get_label_radius(label)

                # compute alpha from score
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

        # pad to nearest multiple of macro_block_size (16) to avoid ffmpeg auto-resize
        w0, h0 = pil.size
        pad_w = ((w0 + 15) // 16) * 16
        pad_h = ((h0 + 15) // 16) * 16
        if pad_w == w0 and pad_h == h0:
            arr = np.array(pil)
        else:
            bg = Image.new('RGB', (pad_w, pad_h), (0, 0, 0))
            left_off = (pad_w - w0) // 2
            top_off = (pad_h - h0) // 2
            bg.paste(pil, (left_off, top_off))
            arr = np.array(bg)
        writer.append_data(arr)
        processed += 1
        if processed % 50 == 0:
            print(f'Wrote {processed}/{len(frames)} frames (last annotation frame {fr} -> video idx {vid_idx})')

    writer.close()
    reader.close()
    print(f'Done: wrote {processed} frames to {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--video', required=True)
    p.add_argument('--ann', required=True)
    p.add_argument('--scored', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--max-frames', type=int, default=None, help='Optional: limit processed annotation frames (quick preview)')
    args = p.parse_args()
    draw_overlay(args.video, args.ann, args.scored, args.out, max_frames=args.max_frames)


if __name__ == '__main__':
    main()
