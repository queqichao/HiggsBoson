import HiggsBosonCompetition

Lr = [0.1, 0.05, 0.02, 0.01]
Results = []
data = HiggsBosonCompetition.read_data("../data/training.csv")
for lr in Lr:
  tmp = []
  Results.append('%.5f' % HiggsBosonCompetition.eval_one_param_adaboost(lr, 200, data));

f = open('../data/res', 'w')

f.write(' '.join(Results))

f.close()
