import HiggsBosonCompetition

Gamma = [1000, 100, 10, 1]
C = [1000000, 10000, 100, 1]
Results = []
data = HiggsBosonCompetition.read_data("../data/training.csv")
partitions = HiggsBosonCompetition.data_partition(len(data.features_), 10)
for gamma in Gamma:
  tmp = []
  for c in C:
    tmp.append('%.5f' % HiggsBosonCompetition.eval_one_param(gamma, c, data, partitions))
  Results.append(tmp)

f = open('../data/res', 'w')

for i in range(len(Results)):
  f.write(' '.join(Results[i])+'\n')

f.close()
