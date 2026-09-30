from pathlib import Path
import json,argparse,hashlib
R=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--prototype',action='store_true');p.add_argument('--variant',choices=['B5','B6']);a=p.parse_args()
template=json.loads((R.parent/'round4/b4/model.ipynb').read_text());base_meta=json.loads((R.parent/'round4/b4/kernel-metadata.json').read_text())
for variant in ([a.variant] if a.variant else ['B5','B6']):
    slug='biohub-'+variant.lower()+'-'+('track-video-runtime' if a.prototype else 'track-video-lineage')
    out=R/(variant.lower()+('_prototype' if a.prototype else ''));out.mkdir(exist_ok=True)
    prelude='''import os,json,hashlib,time,sys,subprocess
from pathlib import Path
NOTEBOOK_STARTED=time.monotonic()
VARIANT=%r
slug='biohub-b5-b6-track-video-models-20260923'
roots=[Path('/kaggle/input/datasets/shawsebastian')/slug,Path('/kaggle/input')/slug]
found=[p for p in roots if (p/'artifact_manifest.json').is_file()]
assert len(found)==1,'Expected private cloud-trained model dataset'
LINEAGE_ROOT=found[0]
manifest=json.loads((LINEAGE_ROOT/'artifact_manifest.json').read_text())
if (LINEAGE_ROOT/'artifact_bundle'/'artifact_manifest.json').exists():
    assert (LINEAGE_ROOT/'artifact_bundle'/'artifact_manifest.json').read_bytes()==(LINEAGE_ROOT/'artifact_manifest.json').read_bytes()
    LINEAGE_ROOT=LINEAGE_ROOT/'artifact_bundle'
if (LINEAGE_ROOT/'artifact_bundle.zip').exists():
    import zipfile
    destination=Path('/kaggle/working/frozen_lineage_artifact');destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(LINEAGE_ROOT/'artifact_bundle.zip') as archive:
        assert set(archive.namelist())==set(manifest['files'])|{'artifact_manifest.json'}
        assert all(Path(n).name==n for n in archive.namelist())
        archive.extractall(destination)
    assert (destination/'artifact_manifest.json').read_bytes()==(LINEAGE_ROOT/'artifact_manifest.json').read_bytes()
    LINEAGE_ROOT=destination
for name,digest in manifest['files'].items():
    if name.endswith('.py') or name=='final_selection.json':assert hashlib.sha256((LINEAGE_ROOT/name).read_bytes()).hexdigest()==digest,name
selection=json.loads((LINEAGE_ROOT/'final_selection.json').read_text())[VARIANT]
for stage in [selection['base'],*selection['stages']]:
    for name in stage['model_files']:assert hashlib.sha256((LINEAGE_ROOT/name).read_bytes()).hexdigest()==manifest['files'][name],name
os.environ.update(BIOHUB_EVENT_ENABLED='0',BIOHUB_VALIDATOR_ENABLE='0',BIOHUB_GRAPH_CACHE='/kaggle/working/reference_graphs')
print('B3/B4 baseline: user-confirmed 0.965. Target >0.990; this notebook makes no verified score claim.')
print('Validation limitations:',manifest['base_checkpoint_training_overlap'])
print('Frozen candidate configuration:',selection)
'''%variant
    if not a.prototype:
        manifest_file=R/'final_reports/artifact_manifest.json'
        digest=hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        prelude+="assert hashlib.sha256((LINEAGE_ROOT/'artifact_manifest.json').read_bytes()).hexdigest()==%r,'Unexpected artifact version'\n"%digest
        prelude+="assert manifest['stage']=='audited_frozen_candidates' and selection.get('calibration_frozen') is True\n"
        prelude+="acceptance=json.loads((LINEAGE_ROOT/'release_acceptance_b56.json').read_text());decision=json.loads((LINEAGE_ROOT/'submission_decision_b56.json').read_text());assert decision['variants'][VARIANT]['submit']\n"
    source=''.join(template['cells'][2]['source']).replace('b4_reference_prediction',variant.lower()+'_reference_prediction')
    post=''.join(template['cells'][3]['source'])
    description=f'''# Biohub {variant} Track Video Lineage

Private cloud training based on the user's B3/B4 pipelines (confirmed public scores 0.965) and the attributed [zhincez reference](https://www.kaggle.com/code/zhincez/biohub-0-947-lb-runnable-with-public-datasets).

The new model encodes raw 3D microscopy along parent and daughter tracklets over ten frames, with symmetric division heads and joint constrained event assignment. Alternative local temporal context, cross-embryo training checks and detector-centered hard-negative ranking are evaluated before selection. The attached selection identifies exactly which trained components are active. The >0.990 leaderboard target is not a verified result.

New training excludes calibration videos, 32 newly reserved audit videos and the historical audit. Inherited detectors and B3/B4 models overlap these sets, so complete-pipeline scores are not fully out-of-fold. Separate randomly initialized cross-embryo models assess new-encoder domain transfer.

All training images, sample banks and checkpoints remain on cloud servers. Inference retains FP32 reference detection with eight D4 transforms and uses dual T4 GPUs. Structural validation precedes atomic CSV export. Attribution: Biohub competition authors, zhincez, Pilkwang, Reyhan Satria, nusrati and evgendvorkin.
'''
    if not a.prototype:
        selected=json.loads((R/'final_reports/final_selection.json').read_text())[variant]
        last=selected['stages'][-1]
        methods={
            'track_set':'Two independently seeded candidate-set attention networks compare competing daughter cells using learned ten-step image tracklets, motion and photometric features. The image encoder and event heads were trained on clean and deliberately corrupted predicted trajectories.',
            'track_rank':'Three seeded LambdaRank models compare daughter candidates within each parent query using learned image-tracklet features, local motion and brightness. Movie-grouped internal validation chooses tree counts and probability temperature.',
            'track_ensemble':'Independently trained candidate-set attention and query-ranking experts combine complementary association judgments over learned image-tracklet and physical-motion features.'}
        method=methods.get(last['kind'],'The attached frozen selection identifies the exact additional trained stage and its configuration.')
        acceptance=json.loads((R/'final_reports/release_acceptance_b56.json').read_text())
        validation_note=('The initial B4-based candidates failed the reserved audit. The revised pipelines preserve B3 divisions and add the fixed visual-link review before the new learned stage, informed by those results; the repeated 32-movie comparison is exploratory and is not independent holdout validation. The original failures and revision protocols are included. The extra edge-component guard remains failed. Submission follows an explicit official-score tradeoff: aggregate scores improve on all three complete movie groups while node recall is preserved; no unseen-score guarantee is made.' if acceptance.get('audit_reused') else 'Two configurations and their weight hashes were frozen before the reserved audit; its complete results are included in the private artifact.')
        description=f'''# Biohub {variant} Frozen Lineage Model

{method} A constrained global assignment changes inherited links under the frozen consensus policy. The selected configuration is `{selected['calibration_report']}/{selected['calibration_arm']}`.

The reference B3/B4 models have user-confirmed public scores of 0.965. The >0.990 target has not been verified. The full pipeline, including inherited models, is not entirely out of fold: inherited detector/model training overlaps validation movies. New fitting excludes 18 calibration movies, 32 reserved audit movies and 18 historical audit movies. {validation_note}

Images, checkpoints and training banks remain in the cloud. Inference uses the original FP32 detector with eight D4 transforms on dual T4 GPUs, verifies artifact hashes, and validates graph/CSV structure before export.

Based on the user's B3/B4 pipelines and the attributed [zhincez reference](https://www.kaggle.com/code/zhincez/biohub-0-947-lb-runnable-with-public-datasets). Attribution: Biohub competition authors, zhincez, Pilkwang, Reyhan Satria, nusrati and evgendvorkin.
'''
    cells=[{'cell_type':'markdown','metadata':{},'source':description.splitlines(True)}]
    for s in [prelude,source,post]:
        compile(s,'notebook_cell','exec');cells.append({'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':s.splitlines(True)})
    notebook={'cells':cells,'metadata':template['metadata'],'nbformat':4,'nbformat_minor':4};(out/'model.ipynb').write_text(json.dumps(notebook,indent=1),encoding='utf-8')
    metadata={**base_meta,'id':'shawsebastian/'+slug,'title':'Biohub '+variant+(' Track Video Runtime' if a.prototype else ' Track Video Lineage')}
    metadata['dataset_sources']=[s for s in metadata['dataset_sources'] if 'biohub-b3-b4-lineage-models' not in s]+['shawsebastian/biohub-b5-b6-track-video-models-20260923']
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2));print('BUILT',metadata['id'])
