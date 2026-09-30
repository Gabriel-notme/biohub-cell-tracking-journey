from pathlib import Path
source=(Path('/workspace/biohub')/'train_centroids.py').read_text()
for old,new in [('Centroid_v1','Centroid_v2'),('Predicted7_v1_frozen.pt','Predicted7_v2_frozen.pt'),('20261401','20261402')]:
    assert old in source;source=source.replace(old,new)
exec(compile(source,'train_centroids_v2','exec'))
