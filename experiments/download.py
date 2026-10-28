"""Download pinned UR3 / FactoryNet inputs; the data remain subject to source terms."""
import argparse
import json
from pathlib import Path
import urllib.request
from experiments.core import sha, dump, stamp
from experiments.prepare import FACTORY_SHA, METADATA_SHA, UR3_SHA
REV='386d731fab807f53ac17a69cf209a2b42b491072'
HF=f'https://huggingface.co/datasets/Forgis/FactoryNet/resolve/{REV}/data/raw/'
FILES={
    'ur3':('https://archive.ics.uci.edu/static/public/963/ur3+cobotops.zip','ur3_source.zip',UR3_SHA),
    'factory':(HF+'factorywave_ur_consolidated.parquet','factorywave_ur_consolidated.parquet',FACTORY_SHA),
    'factory-metadata':(HF+'factorywave_data_episode_metadata.parquet','factorywave_data_episode_metadata.parquet',METADATA_SHA),
}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('dataset',choices=FILES);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    url,name,expected=FILES[a.dataset];a.output.mkdir(parents=True,exist_ok=True);path=a.output/name
    if path.exists():
        if sha(path)!=expected:raise ValueError('Existing download hash mismatch; move it aside before retrying')
        print('Existing pinned file verified:',path);return
    part=path.with_suffix(path.suffix+'.partial')
    if part.exists():raise FileExistsError('A partial attempt already exists; inspect/move it before retrying')
    m=dict(state='downloading',started_at=stamp(),url=url,expected_sha256=expected)
    try:
        with urllib.request.urlopen(url,timeout=60) as response,part.open('xb') as stream:
            m['resolved_url']=response.geturl()
            while block:=response.read(1024*1024):stream.write(block)
        actual=sha(part)
        if actual!=expected:raise ValueError(f'Hash mismatch: {actual}')
        part.replace(path);m.update(state='completed',sha256=actual,bytes=path.stat().st_size,ended_at=stamp())
    except BaseException as e:m.update(state='failed',error=repr(e));raise
    finally:dump(a.output/f'{name}.download.json',m)
    print('Verified:',path)
if __name__=='__main__':main()
