# 2 Qubit RB on the QOP Platform: Introduction and API

## Introduction

This module allows users to implement two qubit randomized benchmarking experiments.

## API

The module contains the following methods:

```python
class TwoQubitRb:
	def __init__(self,
				 config: dict,
				 native_2_qubit_cliffords: dict[str, callable],  # e.g. {'iswap': fun}, {'iswap': fun1, 'cnot12': fun2},... for decomposition, options are: iswap, cz, cnot12, cnot21. Signature must be: (baking_obj)
				 c1_func: callable,  # signature must be: (baking_obj, x, z, a, q_name)
				 prep_func: callable,  # any signature (pass args, kwargs in run)
				 measure_func: callable  # any signature (pass args, kwargs in run)
			 )
	
	 def run(self,
			 sequence_depth: list[int],
			 num_repeats: int,  # this is for different sequences
			 num_averages: int,  # how much to average every sequence
			 interleaving_gate: Optional[Union[CliffordTableau, TwoQubitCliffordDecomposition]],
			 prep_args: list,
			 prep_kwargs: dict,
			 measure_args: list,
			 measure_kwargs: dict
			 ) -> ResultsObj
	
	def analyze_results(raw_results: ResultsObj) -> AnalyzedResultObj


class TwoQubitCliffordDecomposition
	"""
	based on p. 11 here: https://arxiv.org/pdf/1402.4848.pdf
	"""
	def __init__(self,
				 gate_class,  # one of 'id', 'cnot', 'iswap', 'swap',
				 c1_q1_a, c1_q1_x, c1_q1_z,
				 c1_q2_a, c1_q2_x, c1_q2_z,
				 s1_q2_a, s1_q2_x, s1_q2_z,
				 s1_q2_a, s1_q2_x, s1_q2_z
				 )
			 
```
Example:
```python
def cz(baking_obj):
	baking_obj.play("c_z", "flux1")

def c1(baking_obj, x, z, a, q_name):
	baking_obj.frame_rotation(z+a, f'xy_{q_name}')
	baking_obj.play("x"*amp(x), f'xy_{q_name}')
	baking_obj.frame_rotation(-a, f'xy_{q_name}')

def prep(time):
	wait(time)

def measure():
	I1 = declare(fixed)
	I2 = declare(fixed)
	measure("meas", "rr1", None, dual_demod.full("cos_iw", "o1", "sin_iw", "o2", I1))	
	measure("meas", "rr2", None, dual_demod.full("cos_iw", "o1", "sin_iw", "o2", I2))
	return I1 > 0, I2 > 0
	

rb = two_qubit_rb(config, {'cz': cz}, c1, prep, measure)

results = rb.run([20, 30, 40], 100, 100
				 prep_args=[10000],
				 prep_kwargs={},
				 measure_args=[],
				 measure_kwargs={}
				 )

analysis_results = rb.analyze(results) 	
```

## Implementation overview

The QUA program is implemented by creating $192=24+24\times3+24\times3+24$ cases in a switch-case construction for each underlying quantum element. Each one of these cases corresponds to a different *single qubit* implementation of one of the classes in p. 11 of https://arxiv.org/pdf/1402.4848.pdf. We "bake" the complete sequence for each one of the cases above as a pulse for each of the underlying quantum element. Each quantum element receives the case indepedently of the others, and they all start playing together.


