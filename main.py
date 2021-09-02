import numpy as np
import csv
import pandas as pd
import random
import math
from copy import deepcopy

from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


def dados(data, share, treinamento=[], teste=[]):
        dataset = list(data)

        # seleciona os dados de dentro da base de dados
        # share define a proporção entre o conjunto de treinamento e o de teste

        for x in range(len(dataset)-1):
                # print(x, "(x)", dataset[x])
                #caso aleatório
                if random.random() < share:
                    treinamento.append(dataset[x])
                else:
                    teste.append(dataset[x])

                #caso teste
                # if x % 50 > 15:
                #     treinamento.append(dataset[x])
                # else:
                #     teste.append(dataset[x])
        print(len(treinamento), " treinamento", len(teste), " teste")

def randomsubset(elementos, train=[]):
    #retorna subconjunto do conjunto original de treinamento com numero especifico de elementos
    resp = []
    i = 0
    # print(elementos,'nn', len(train))
    while len(resp) < elementos:
        if random.random() < 0.8:
            resp.append(train[i])

        i = i+1

    return resp

def correct(datamatrix=[], corrector=[]):
    # target = []
    for x in range(len(datamatrix)):
        #analisa uma linha da matriz, infos de um cadidato
        for y in range(len(datamatrix[x])):
            #y analisa feature por feature
            if not isinstance(corrector[y], str):
                # print(corrector[y])
                aux = []
                aux = list(corrector[y])
                if y == 1:
                    #city
                    if datamatrix[x][y] in aux:
                        datamatrix[x][y] = 1
                    else:
                        datamatrix[x][y] = 0

                elif y==8 or y ==11:
                    #experience e last new job
                    if datamatrix[x][y] in aux:
                        if aux.index(datamatrix[x][y])==0:
                            datamatrix[x][y] = aux[-1]
                        else:
                            datamatrix[x][y] = aux[-2]
                    else:
                        if datamatrix[x][y] == '':
                            datamatrix[x][y] = 0
                        else:
                            datamatrix[x][y] = int(datamatrix[x][y])
                        if y == 8:
                            if datamatrix[x][y] < 5:
                                datamatrix[x][y] = 1
                            elif datamatrix[x][y] < 11:
                                datamatrix[x][y] = 2
                            else:
                                datamatrix[x][y] = 3

                else:
                    # todos os outros casos com strings: gender, relevant_experience, enrolled university, education level, major discipline, company size, compary type
                    datamatrix[x][y] = aux.index(datamatrix[x][y])

            else:
                #id, idh, training, answer
                datamatrix[x][y] = float(datamatrix[x][y])
                if y == 2:
                    #idh
                    if datamatrix[x][y] > 0.7:
                        datamatrix[x][y] = 1
                    else:
                        datamatrix[x][y] = 0
                elif y == 12:
                  #training hours
                    aux = [50,150,250]
                    if datamatrix[x][y] < aux[0]:
                        datamatrix[x][y] = 0
                        #menos de 50h
                    elif datamatrix[x][y] < aux[1]:
                        datamatrix[x][y] = 1
                        #de 50 a 149h de treinamento
                    elif datamatrix[x][y] < aux[2]:
                        datamatrix[x][y] = 2
                        #de 150 a 249h de treinamento
                    else:
                        datamatrix[x][y] = 3
                        #acima de 250h
                # elif y == 13:
                #     #tira o último elemento pra separar em data e target. Vai que dá
                #     target.append(datamatrix[x][y])
                #     datamatrix[x].pop()

    # return target

def target(mat=[]):
    alvo = []
    # convert = ['0.0', '1.0']
    for x in range(len(mat)):
        # tira o último elemento pra separar em data e target. Vai que dá
        # inti = convert.index(mat[x][-1])
        # print(inti, "and", mat[x][-1])
        alvo.append(mat[x][-1])
        mat[x].pop()

    return alvo


def idOut(mat=[]):
    saveourid = []
    for x in range(len(mat)):
        saveourid.append(mat[x][0])
        mat[x].pop(0)

    return saveourid

def pdont(mat=[]):

    #pandas are crashing me
    for x in range(len(mat)):
        del mat[x][7]
        del mat[x][4]
        del mat[x][3]
        del mat[x][1]
        del mat[x][0] #já em idout


# --- main -----------
# a = pd.read_csv("aug_train.txt")
# a = pd.DataFrame(a)
# print(a)

dado = csv.reader(open("aug_train.csv", "r"))
todosDados = list(dado)
labels = todosDados[0]
print(labels)
nomes = deepcopy(todosDados[0])
todosDados.pop(0) #tira as labels

fulldataframe = pd.DataFrame(todosDados, columns=labels)

#analisando os dados a priori, pode-se tirar várias conclusões
#temos 14 colunas de dados. Vamos trabalhar efetivamente com 12
#A primeira é apenas o ID do candidato. Não faz sentido ser usada pra calcular. A última é a resposta: se ele está ou não disposto a mudar de emprego
#As 12 centrais são importantes. Dessas, apenas duas não precisam de nenhum tratamento: IDH e Horas de treinamento
# Horas de treinamento será alocados em faixas: [0-49, 50-149, 150-249, 250+].
# IDH vai ser 0 pra abaixo de 0.7 e 1 pra acima
#Das outras 10: gênero e se tem experiência na área são binárias, podem ser substituídos por 0s e 1s
# Haviam 4 cidades que tinham mais aplicantes quetodo o resto. Serão consideradas um grupo prioritário (feito uma região metropolitana)
# Experience precisa consertar <1 e >20. Last new job precisa tirar o >4 e never
# e todas as outras tem diversos valores que vão ser convertidos em numéricos, só pq é mais fácil para o algoritmo tratar

unicos = labels
unicos[1] = ["city_103", "city_21", "city_16", "city_114"]
unicos[3] = fulldataframe['gender'].unique()
unicos[4] = fulldataframe['relevent_experience'].unique()
unicos[5] = fulldataframe['enrolled_university'].unique()
unicos[6] = fulldataframe['education_level'].unique()
unicos[7] = fulldataframe['major_discipline'].unique()
unicos[9] = fulldataframe['company_size'].unique()
unicos[10] = fulldataframe['company_type'].unique()

unicos[8] = ['<1', '>20',4,0] #experience. Valores tb colocados em faixas: 1-4:1; 5-10:2;11-20:3
unicos[11] = ['never', '>4', 5, 0] #last new job
#depois eu vi que numpy faz o que eu fiz na mão em únicos kkkk dps eu testo isso


treinamento =[]
teste = []
# tamanhotreino = [150, 500, 1000, 2000, 5000]
# tamanhoteste = [50, 100, 250, 500, 1000]
tamanhotreino = [2000, 5000, 10000]
tamanhoteste = [500, 1000, 3000]
listaTreino = []
listaTeste = []

dados(todosDados, 0.7, treinamento, teste)

#gerar subsets do tamanho especificado aqui em cima e tratá-los
# for x in range(len(tamanhoteste)):
#
#     t2 = []
#     t2 = randomsubset(tamanhotreino[x], treinamento)
#     t = deepcopy(t2)
#     # print("treinamento", x)
#     correct(t, unicos)
#     listaTreino.append(t)
#     t3 = []
#     t3 =randomsubset(tamanhoteste[x], teste)
#     # print("teste",x)
#     t = deepcopy(t3)
#     correct(t, unicos)
#     listaTeste.append(t)

#Testa com caso pedido pela tarefa no kaggle:
testefinal = csv.reader(open("aug_test.csv", "r"))
grandeTeste = list(testefinal)
grandeTeste.pop(0) #tira as labels
print("Começa teste final:", '\n')
tf = grandeTeste
correct(tf, unicos)
pdont(tf)
# print(tf[0])
print(nomes)
del nomes[-1]
del nomes[7]
del nomes[4]
del nomes[3]
del nomes[1]
del nomes[0]
ytf = pd.DataFrame(tf, columns=nomes)
# print(ytf)
#to feature analyis
# nomes.pop(0)
# nomes.pop()

#seta loopfinal
#batch 1
# hiddenlist = [2, 5, 10]
# epocas = [1000, 2000, 5000]
# arvores = [200, 500, 1000]
#batch2
# hiddenlist = [2, 5]
# epocas = [2000, 5000]
# arvores = [100, 2000]
hiddenlist = [5]
epocas = [5000]
arvores = [250]

mlpResults = []
rfResults = []
kagglemlp = []
kagglerf = []
smotemlp =[]
smoterf =[]

#testefinal
correct(treinamento, unicos)
correct(teste, unicos)
listaTeste.append(teste)
listaTreino.append(treinamento)

k=4
seed = 100
sm = SMOTE(sampling_strategy='auto', k_neighbors=k, random_state=seed)


#------------------------------- bora lá --------------------------------------------------------------
for x in range(len(listaTeste)):

    print("lista: ", x)

    treine = listaTreino[x]
    testes = listaTeste[x]
    pdont(treine)
    pdont(testes)

    # correct(treine, unicos)
    y2 = target(treine)
    print("% Positivos de Treinamento:", sum(y2)/len(y2))
    # id2 = idOut(treine)

    # plt.title('dataset original')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()

    # correct(testes, unicos)
    y3 = target(testes)
    # id3 = idOut(testes)
    print("% Positivos de Teste:", sum(y3) / len(y3))

    scores = []

    smoteTrainX, smoteTrainY = sm.fit_resample(treine, y2)
    smoteTestX, smoteTestY = sm.fit_resample(teste, y3)
    print("% Positivos de Treinamento:", sum(smoteTrainY) / len(smoteTrainY))
    print("% Positivos de Treinamento:", sum(smoteTestY) / len(smoteTestY))

    for y in range(len(epocas)):
        # treina
        # 1: MLP
        hidden = hiddenlist[y]
        mlpEpocas = epocas[y]
        mlp = MLPClassifier(solver='sgd', hidden_layer_sizes=(hidden, 2), learning_rate='constant',
                            learning_rate_init=0.01, max_iter=mlpEpocas, activation='relu')
        mlp.fit(treine, y2)
        mlp.fit(smoteTrainX, smoteTrainY)

        #testa e valida
        previsao = mlp.predict(testes)
        mlpaccuracy = accuracy_score(y3, previsao)*100
        # previsao = mlp.predict(smoteTestX)
        # mlpaccuracy = accuracy_score(smoteTestY, previsao) * 100
        scores.append(mlpaccuracy)
        cm =confusion_matrix(y3, previsao, normalize="all")
        # cm =confusion_matrix(smoteTestY, previsao, normalize="all")
        print("mlp: ", y,'at:', mlpaccuracy,  '\n', cm)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        # disp.plot()

        mlpredict = mlp.predict(tf)

        kagglemlp.append(mlpredict)

        mlp.fit(smoteTrainX, smoteTrainY)
        previsao = mlp.predict(smoteTestX)
        mlpaccuracy = accuracy_score(smoteTestY, previsao) * 100
        cm = confusion_matrix(smoteTestY, previsao, normalize="all")
        print("Smote mlp: ", y, 'at:', mlpaccuracy, '\n', cm)
        smotemlp.append((mlp.predict(tf)))


    mlpResults.append(scores)
    scores =[]

    for z in range(len(arvores)):
        #2: RF
        rf = RandomForestClassifier(n_estimators=arvores[z])
        rf.fit(treine, y2)
        # rf.fit(smoteTrainX, smoteTrainY)

        preveja = rf.predict(testes)
        rfaccuracy = accuracy_score(y3, preveja) * 100

        # preveja = rf.predict(smoteTestX)
        # rfaccuracy = accuracy_score(smoteTestY, preveja) * 100

        scores.append(rfaccuracy)

        cn = confusion_matrix(y3, preveja, normalize="all")
        # cn = confusion_matrix(smoteTestY, preveja, normalize="all")
        print("rf: ", z, 'at:', rfaccuracy, '\n', cn)
        # displ = ConfusionMatrixDisplay(confusion_matrix=cn)
        # displ.plot()
        rfpredict = rf.predict(tf)
        kagglerf.append(rfpredict)

        # impfeat = pd.Series(rf.feature_importances_, index=nomes).sort_values(ascending=False)
        # print('floresta: ', z,'\n', impfeat)
        #smote
        rf.fit(smoteTrainX, smoteTrainY)
        preveja = rf.predict(smoteTestX)
        rfaccuracy = accuracy_score(smoteTestY, preveja) * 100
        cn = confusion_matrix(smoteTestY, preveja, normalize="all")
        print("smote rf: ", z, 'at:', rfaccuracy, '\n', cn)
        smoterf.append(rf.predict(tf))

    rfResults.append(scores)

# print("MLP Results")
# print(mlpResults)
# print("RF", '\n',rfResults)

# print(list(kagglemlp[0]))
# print(list(kagglerf[0]))

fsmlp =[]
fsrf = []
smotemlpresults =[]
smoterfresults = []
print(len(kagglemlp[0]),'len', len(smotemlp[0]), "smotelen")
for a in range(len(kagglemlp)):
    # print(len(kagglemlp))
    # print(len(kagglerf))
    # print("finalmente")
    # scoremlp = 100*sum(list(kagglemlp[a]))/len(tf)
    # scorerf = 100*sum(list(kagglerf[a]))/len(tf)
    scoremlp = sum(list(kagglemlp[a]))
    scorerf = sum(list(kagglerf[a]))
    smotemlpresults.append(sum(list(smotemlp[a])))
    smoterfresults.append(sum(list(smoterf[a])))


    fsmlp.append(scoremlp)
    fsrf.append(scorerf)

print("mlp", fsmlp,'\n', "rf", fsrf)
print("Smote MLP:", smotemlpresults, '\n', "Smote RF: ", smoterfresults)






