"""Emit a corrected LOCAL RESPONSE qualification; never submits a cloud job.

The probe is +/- RZ(pi/2) on qubit 0; readout is Z_0. Their half-difference
measures i<[Z_0,Z_0(t)]>/2 exactly, without an ancilla, unpreparation, or a
global survival estimator. This is a commutator response, NOT |L(t)|^2,
not a positive spectral density and not an equilibrium fluctuation spectrum.
"""
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import native_response as N

def z(p,q):
    idx=np.arange(len(p)); return float(np.dot(abs(p)**2,1-2*((idx>>q)&1)))

def main():
    out=HERE/'response_v6'; out.mkdir(exist_ok=True)
    pairs=[(0,3),(1,4),(2,5)]; gs=N.run(6,N.prep(pairs))
    rows=[]; calibration=[]
    for name,extra in N.FAMILIES.items():
        H=N.hamiltonian(6,pairs,extra)
        w,V=np.linalg.eigh(H)
        for t in [0.,0.125,0.25,0.5,1.]:
            for sign in [-1,1]:
                probe=[('rz',0,sign*math.pi/2)]
                initial=N.run(6,probe,gs)
                exact=V@(np.exp(-1j*w*t)*(V.T@initial)); chosen=None
                for steps in [1,2,4,8,16,32,64]:
                    ops=N.prep(pairs)+probe+N.evolution(pairs,extra,t,steps)
                    p=N.run(6,ops)
                    inf=max(0.,float(1-abs(np.vdot(exact,p))**2))
                    obs_err=abs(z(p,0)-z(exact,0))
                    if inf<1e-4 and obs_err<0.005:
                        chosen=(steps,ops,p,inf,obs_err); break
                if chosen is None: raise RuntimeError('No converged circuit; do not emit')
                steps,ops,p,inf,obs_err=chosen
                label=f'{name}_t{t:g}_s{sign:+d}'
                text=N.qasm(6,ops); file=label+'.qasm'; (out/file).write_text(text,newline='\n')
                rows.append({'id':label,'family':name,'t':t,'sign':sign,'file':file,
                             'sha256':hashlib.sha256(text.encode()).hexdigest(),
                             'steps':steps,'readout':'Z0','expected_circuit':z(p,0),
                             'expected_continuous':z(exact,0),'state_infidelity':inf,
                             'observable_discretization_error':obs_err,**N.cost(ops)})
            if name=='null':
                # Independent spectral prediction from the four-level pair.
                chi=math.atan(N.DELTA/2)
                pred=-(math.cos(chi)**2*math.sin(N.DELTA*t)
                       +math.sin(chi)**2*math.sin((N.KAPPA+N.DELTA)*t))
                measured=(rows[-1]['expected_continuous']-rows[-2]['expected_continuous'])/2
                calibration.append({'t':t,'predicted':pred,'computed':measured,'error':abs(pred-measured)})
    assert max(r['error'] for r in calibration)<1e-10
    # Thresholds set now for a future job. No previously flown counts are graded as v6.
    manifest={'version':6,'status':'MODEL_VALIDATED_NOT_FLOWN','n_qubits':6,
              'shots_per_circuit':4096,'circuits':rows,'calibration':calibration,
              'observable':'half difference of Z0 after +/- pi/2 Z0 probe',
              'hardware_gates':{'max_abs_observable_error':0.08,'familywise_alpha':0.01,
                                'replicate_required':True},
              'total_shots_per_run':4096*len(rows),
              'backend':None,'physical_layout':None,'cost_currency':None,
              'claim':'local-response qualification; not AML accuracy or computational advantage',
              'provenance':'constructed from kappa, local coupled pair and payment edges; no fitted eigengap in circuit',
              'not_a_frequency_resolution_experiment':True}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps({'circuits':len(rows),'shots':manifest['total_shots_per_run'],
                      'max_logical_cx':max(x['logical_cx_before_routing'] for x in rows),
                      'calibration_error':max(x['error'] for x in calibration)},indent=2))

if __name__=='__main__': main()
