"""Add the dense-probability save patch to a built P-kernel notebook (kernel dir given), in the FULLGRAPH patch cell."""
import json, sys
kdir = sys.argv[1]
nbp = kdir + '/model.ipynb'
PATCH = r'''
os.environ['DENSE_DIR'] = '/kaggle/working/dense'
_dn_src = _batch_path.read_text()
_dn_a1 = "        coords, edges = predict_video(\n"
assert _dn_src.count(_dn_a1) == 1, 'dense anchor 1'
_dn_src = _dn_src.replace(_dn_a1, "        globals()['_DENSE_ACC'] = []\n" + _dn_a1)
_dn_a2 = "            candidates = sorted(\n"
assert _dn_src.count(_dn_a2) == 1, 'dense anchor 2'
_dn_src = _dn_src.replace(_dn_a2, "            if os.environ.get('DENSE_DIR'):\n                _dn_ii, _dn_jj = np.nonzero(probs > float(os.environ.get('DENSE_MIN', '0.02')))\n                globals().setdefault('_DENSE_ACC', []).append(np.stack([np.asarray(idx_src)[_dn_ii], np.asarray(idx_tgt)[_dn_jj], np.round(probs[_dn_ii, _dn_jj] * 1e6).astype(np.int64)], 1))\n" + _dn_a2)
_dn_a3 = "            save_graph(graph, _fgd / f'{name}.geff')\n"
assert _dn_src.count(_dn_a3) == 1, 'dense anchor 3'
_dn_src = _dn_src.replace(_dn_a3, _dn_a3 + "        if os.environ.get('DENSE_DIR') and globals().get('_DENSE_ACC'):\n            _dn_d = Path(os.environ['DENSE_DIR']); _dn_d.mkdir(parents=True, exist_ok=True)\n            np.savez_compressed(_dn_d / f'{name}.npz', edges=np.concatenate(globals()['_DENSE_ACC']), coords=np.asarray(coords, dtype=np.float64))\n")
_batch_path.write_text(_dn_src)
print('DENSE patch installed', flush=True)
'''
nb = json.load(open(nbp))
anchor = "print('FULLGRAPH patch installed', flush=True)\n"
n = 0
for c in nb['cells']:
    if c['cell_type'] != 'code': continue
    s = ''.join(c['source'])
    if s.count(anchor) == 1 and 'DENSE patch installed' not in s:
        c['source'] = [s.replace(anchor, anchor + PATCH)]; n += 1
assert n == 1, n
json.dump(nb, open(nbp, 'w'), indent=1)
print('dense patch added to', nbp)
