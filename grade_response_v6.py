"""Strict offline hardware qualification with simultaneous Hoeffding bounds.
Usage: python grade_response_v6.py receipt.json
Receipt: {manifest_sha256, job_id, backend, physical_layout, counts:{id:{bitstring:integer}}}.
All measurement strings use QASM classical order c[n-1]...c[0].
"""
import hashlib,json,math,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def grade(receipt):
    p=HERE/'response_v6'/'manifest.json'; raw=p.read_bytes(); m=json.loads(raw)
    if receipt.get('manifest_sha256')!=hashlib.sha256(raw).hexdigest():
        raise ValueError('receipt not bound to this frozen manifest')
    for field in ['job_id','backend','physical_layout']:
        if not receipt.get(field): raise ValueError('missing '+field)
    if set(receipt['counts'])!={x['id'] for x in m['circuits']}:
        raise ValueError('missing or extra circuit IDs')
    rows=[]; alpha=m['hardware_gates']['familywise_alpha']; M=len(m['circuits'])
    for c in m['circuits']:
        qasm=(HERE/'response_v6'/c['file']).read_bytes()
        if hashlib.sha256(qasm).hexdigest()!=c['sha256']: raise ValueError('modified circuit')
        counts=receipt['counts'][c['id']]
        for b,k in counts.items():
            if len(b)!=6 or any(x not in '01' for x in b) or type(k) is not int or k<0:
                raise ValueError('invalid count entry')
        n=sum(counts.values())
        if n<m['shots_per_circuit']: raise ValueError('insufficient shots')
        estimate=sum((1-2*int(b[-1]))*v for b,v in counts.items())/n
        bound=math.sqrt(2*math.log(2*M/alpha)/n)
        error=abs(estimate-c['expected_circuit'])
        # Require the whole confidence interval to fit the tolerance.
        passed=error+bound<=m['hardware_gates']['max_abs_observable_error']
        rows.append({'id':c['id'],'mean':estimate,'simultaneous_radius':bound,'abs_error':error,'pass':passed})
    return {'job_id':receipt['job_id'],'qualification_pass':all(x['pass'] for x in rows),
            'advantage_demonstrated':False,'independent_replication_required':True,'circuits':rows}

if __name__=='__main__':
    if len(sys.argv)!=2: raise SystemExit('Supply an actual hardware receipt JSON; no synthetic default')
    result=grade(json.loads(Path(sys.argv[1]).read_text()))
    print(json.dumps(result,indent=2)); raise SystemExit(0 if result['qualification_pass'] else 2)
