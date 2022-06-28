import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

max_epocas = int(input("Número máximo de épocas (Recomendao: 50): "))
porcentagem_teste = int(input("Porcentagem de casos de teste (Recomendada: 20) [0-100] : "))/100
porcentagem_validacao = int(input("Porcentagem reservada para validação (Recomendada: 30) [0-100]: "))/100
neuronios_escondidos = int(input("Número de neurônios na camada escondida (Recomendado: 10): "))
max_falha_validacao = int(input("Falhas de validação consecutivas (Recomendação: 6): "))
func_ativacao = input("Função de Ativação (Recomendada: logistic) [identity, logistic, tanh, relu]: ")
taxa_aprendizado = float(input("Taxa de aprendizado (Recomendada: 0.25): "))

def decoder(entrada):
	entrada_decod = []

	for i in entrada:
		if(i[0]==1):
			entrada_decod.append(0)
		elif(i[1]==1):
			entrada_decod.append(1)
		else:
			entrada_decod.append(2)

	return entrada_decod

#Abrindo dados de entrada e formatando para o treinamento
dados = open("DATA.txt", "r")

X1 = []
X2 = []

for linha in dados:
	x1, x2 = linha.split(',')
	X1.append(float(x1))
	X2.append(float(x2))

dados.close()

X = np.column_stack((X1, X2))

#Abrindo os Alvos e formatando para o treinamento
alvos = open("TARGETS.txt", "r")

Y1 = []
Y2 = []
Y3 = []

for linha in alvos:
	y1, y2, y3 = linha.split(',')
	Y1.append(int(y1))
	Y2.append(int(y2))
	Y3.append(int(y3))

alvos.close()

T = np.column_stack((Y1, Y2, Y3))

#Divisão treino/teste
x_treino, x_teste, t_treino, t_teste = train_test_split(X, T,
	test_size=porcentagem_teste, random_state = 100)

#Divisão treino/validação
x_treino, x_valid, t_treino, t_valid = train_test_split(x_treino, t_treino,
	test_size=porcentagem_validacao, random_state=100)

clf = MLPClassifier(hidden_layer_sizes=(neuronios_escondidos,),
	activation=func_ativacao, validation_fraction=0,random_state=100,
	learning_rate_init = taxa_aprendizado)

early_stopping = 0
ultimo_acerto_validacao = 0

historico_validacao = []
historico_treino = []

for i in range (0,max_epocas):

	#Se a a validação não melhorar por max_falha_validacao épocas
	if(early_stopping >= max_falha_validacao):
		break

	clf.partial_fit(x_treino, t_treino, [0, 1, 2])

	acerto_treino = clf.score(x_treino, t_treino)
	acerto_validacao = clf.score(x_valid, t_valid)

	historico_treino.append(acerto_treino)
	historico_validacao.append(acerto_validacao)

	if(acerto_validacao <= ultimo_acerto_validacao):
		early_stopping = early_stopping + 1
	else:
		early_stopping = 0

	ultimo_acerto_validacao = acerto_validacao

print("\nO treinamento terminou na época %d" %(i+1))
print("Precisão nos casos de treino: %.2f" %(acerto_treino*100))
print("Precisão na validação: %.2f" %(acerto_validacao*100))
print("Precisão nos casos de teste: %.2f" % (clf.score(x_teste, t_teste)*100))

teste_pred = clf.predict(x_teste)

historico_treino = [(1-i)*100 for i in historico_treino]
historico_validacao = [(1-i)*100 for i in historico_validacao]

plt.plot(historico_treino, label="Treino")

plt.plot(historico_validacao, label="Validação")
plt.xlabel("Época")
plt.ylabel("Erros(%)")
plt.legend()


t_teste_decod= decoder(t_teste)

pred_decod = decoder(teste_pred)

t_teste_decod = np.array(decoder(t_teste)).reshape(1,-1).flatten()

h = .02
plt1, ax1 = plt.subplots(figsize=(10, 16))

#Plota os casos de teste
scatter = ax1.scatter(x_teste[:,0], x_teste[:,1], c = t_teste_decod)
ax1.legend(*scatter.legend_elements(), loc="lower left", title="Classes")

#Criando as fronteiras de decisão
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

#Mapeamento das classes para o plot da fronteira
z = decoder(Z)

Z = np.array(z).reshape(xx.shape)
ax1.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=.8)

#Matriz de confusão
cm = confusion_matrix(t_teste_decod, pred_decod,labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
fig, ax = plt.subplots(figsize=(12,10))
disp.plot(ax=ax)
disp.ax_.set(title='Matriz de Confusão', xlabel='Classe Prevista', ylabel='Classe Alvo')

plt.show()