#:set architecture = [('Gemm', ['input', 1], None), ('Sigmoid', ['input'], [2]), ('Gemm', ['input', 1], None), ('Sigmoid', ['input'], [2]), ('Gemm', ['input', 1], None)]
#:set inputs = []
#:set trueInputs = [['input', [1, 3]]]
#:set outShape = [['output', [1, 3]]]
#:set outputs = {'output': 'input'}
