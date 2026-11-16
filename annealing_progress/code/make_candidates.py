"""Optional proposal generation; not part of the trusted verifier."""
import json
from pathlib import Path
from model import A,MU,run,schedules

result={}
for label in range(3):
    _,path=run((MU[label]@A.T)[None,:],schedules()['middle32'],trajectory=True)
    rows=path[:13,0,:] if label==2 else path[:,0,:]
    result[str(label)]=[[format(float(x),'.16f') for x in row] for row in rows]
path=Path(__file__).resolve().parents[1]/'results/candidates.json'
path.write_text(json.dumps(result,indent=2)+'\n')
print(path)
