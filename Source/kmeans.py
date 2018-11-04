

import math
import random
import csv
from collections import Counter

from scipy import spatial

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Referencia principal para design do algoritmo 
# https://gist.github.com/iandanforth/5862470


def main():
 
    # numero minimo e maximo dos k que queremos avaliar 
    num_min_clusters = 2
    num_max_clusters = 4
    
    #arquivo a ser lido
    arquivo = 'tudo.csv'
    
    
    # @cutoff: Valor para indicar que o algoritmo convergiu para uma resposta
    cutoff = 0.0000001
    
    # @num_iteracoes: Numero de iteracoe
    num_iteracoes = 1000
    
    tipo_dist = 'euclidiana'
    
    erros = inicio_com_elbow( num_min_clusters, num_max_clusters + 1, arquivo, tipo_dist, cutoff, num_iteracoes) 
    
    
    ''' Para x-means'''
    k = num_min_clusters
    taxa_erro_convergencia = erros[num_min_clusters]
    for index_erro in range(num_min_clusters, num_max_clusters):
        print(" - index: ", index_erro)
        print(" - erro: ", erros[index_erro])
        print(erros[index_erro + 1]/erros[index_erro])
        if (erros[index_erro + 1]/erros[index_erro] < taxa_erro_convergencia ):
            taxa_erro_convergencia = erros[index_erro + 1]/erros[index_erro]
            k = index_erro + 1
            

   
    #Roda o algoritmo para o k escolhido, sem desenhar o cotovelo e com parametrizacao rigorosa
    print("K a ser usado: ", k)
    iniciaProcesso(k, arquivo, tipo_dist, cutoff/100, num_iteracoes*5)
    
def inicio_com_elbow(num_min_clusters, num_max_clusters, arquivo, tipo_dist, cutoff, num_iteracoes):
    #inicializando com valor conhecido para ignonarmos posicoes nao desejadas
    erros = [-66666.6] * num_max_clusters
    #K means simples
    for k in range(num_min_clusters, num_max_clusters):
        erros[k] = iniciaProcesso(k, arquivo, tipo_dist, cutoff, num_iteracoes)
        
        
    ''' Grafico Elbow Method'''  
    # Metodo do Cotovelo - Elbow Method
    
    plt.figure(figsize=(12, 12), dpi=80)
    plt.ylabel("Erro")
    plt.xlabel("k-clusters")
    plt.title("Metodo do Cotovelo - Elbow Method")
    
    # Ignorando na posicao minima dos clusters pra frente: num_min_clusters
    plt.plot(range(num_min_clusters, num_max_clusters) ,erros[num_min_clusters:])
    elbow = arquivo.replace(".csv", "_") + tipo_dist + "_elbow.png"
   
    plt.savefig(elbow)

    plt.show()
    return erros
    
def iniciaProcesso(num_clusters, arquivo, tipo_dist, cutoff, num_iteracoes):
    # Lendo arquivo .csv
    pontos = []
    palavras = [""]
    # Cada linha e' um documento e sera convertido em uma lista de Pontos.
    # Cada ponto vai conter a linha toda, convertendo ja para float
    with open(arquivo, newline='') as arq_csv:
        # lendo primeira linha de palavras
        prim_linha = next(arq_csv)
        prim_linha = prim_linha.replace("\r\n", "")
        prim_linha = prim_linha.replace("\"", "")
        palavras = prim_linha.split(',')
        # Imprime numero de dimensoes dos pontos
        #print("LOG: TAMANHO = ", len(palavras) - 1)
        dimensoes = len(palavras) - 1
        spamreader = csv.reader(arq_csv, delimiter=',')
        for row in spamreader:
            documento = row[0]
            del row[0]
            x=0
            # Percorre todas as frequencias
            while (x < dimensoes):
                row[x] = float(row[x])
                x += 1
            p = Ponto(row, documento)
            pontos.append(p)

    # Chamando o algoritmo 
    print ("LOG: Inicializando Kmeans simples...")

    melhor_kmeans, erro = kmeans_simples(num_clusters, pontos, tipo_dist, cutoff, num_iteracoes)
    
    
    print("LOG: Comecando silhouette")
    setosa  = []
    centros = []
    versicolor = []
    virginica = []
    silhouette_x = []
    silhouette_y = []
    silhouette_cluster = []
    silhouettes_medios = []
    palavras_no_cluster = [[] for _ in melhor_kmeans]

    for i, cluster in enumerate(melhor_kmeans):
        #print ("\n---- Cluster ", i, ":")
        media_silhouette_do_cluster = -1
        acumulado_silhouette = 0
        x = 0
        s = 0
        ve = 0
        vi = 0
        centros.append(cluster.centroid.coordenadas)
        for doc in cluster.pontos:
          
            if doc.documento == 'Iris-setosa':
                setosa.append(doc.coordenadas)
                s = s + 1
                
            if doc.documento == 'Iris-versicolor':
                versicolor.append(doc.coordenadas)
                ve = ve + 1

            if doc.documento == 'Iris-virginica':
                virginica.append(doc.coordenadas)
                vi = vi + 1

            
        
            x += 1
            # Registrando palavras de cada Cluster para usar na WordCloud
            palavras_no_cluster[i].append(doc.coordenadas)
            
            
            doc.meuCluster = i
            
            # Descomente para imprimir palavra mais frequente de cada doc por cluster:
            #maior_freq = max(doc.coordenadas)
            #palavra_mais_index =  doc.coordenadas.index(maior_freq)
            #print ("Documento : ", doc, " + freq: ", palavras[palavra_mais_index + 1])
            
            
            
    #Silhouette parte 1
#            # Calculando o Silhouette score
            segundo_mais_proximo = getSegundoClusterMaisProximo(doc, cluster, melhor_kmeans)
            #print(cluster.pontos, "segundo mais perto:", segundo_mais_proximo.pontos)
            doc.silhouette_score = getSilhouetteScore(doc, cluster.pontos, segundo_mais_proximo.pontos)
            acumulado_silhouette += doc.silhouette_score
            ''' para bases menores descomentar o abaixo '''    
            silhouette_x.append(doc.silhouette_score)
            silhouette_y.append(doc)
            ''' fim do descomentar para bases menores '''
            
            silhouette_cluster.append(doc.meuCluster)
        if(x > 0):
            media_silhouette_do_cluster = acumulado_silhouette / x
        silhouettes_medios.append(media_silhouette_do_cluster)
        print("Cluster")
        print("total" + str(s+ve+vi))
        print("Setosa" + str(s))
        print("Versicola" + str(ve))
        print("virginica" +  str(vi))
    
        ''' para bases GRANDES descomentar abaixo '''
#        silhouette_x.append(media_silhouette_do_cluster)
#        silhouette_y.append(i)
        ''' fim do descomentar para bases MAIORES '''

            
    print("LOG: inicializando escrever arquivos")
            
    total_palavras_por_cluster = []
    for k, pontos in enumerate(palavras_no_cluster):
        soma_coord = [0] * dimensoes
        for ponto in pontos:
    
            for i, coord in enumerate(ponto):
                soma_coord[i] += coord
                #print(soma_coord)
        total_palavras_por_cluster.append(soma_coord) 
    
    ''' Arquivo 1: para fazer uma wordcloud'''
    #Escrevendo para fazermos as wordclouds, cada cluster e quantas palavras tem
    output_name = arquivo.replace(".csv", "_") + str(num_clusters) + '_clusters.csv' 

    with open(output_name, 'w',) as csvfile:
        writer = csv.writer(csvfile,  lineterminator='\n')
        writer.writerow(palavras[1:])
        for a in total_palavras_por_cluster:
            writer.writerow(a)
    print("LOG: escrito cluster para wordclouds")
            
    output_name2 = arquivo.replace(".csv", "_") + str(num_clusters) + 'pontos_por_cluster.csv' 
    
    ''' Arquivo 2: pontos por cluster, silhouette medio e erro'''    
    # Arquivo que mostra quais pontos estao em quais cluster + o silhouette medio
    # cluster e o erro
    with open(output_name2, 'w',) as csvfile:
        writer = csv.writer(csvfile,  lineterminator='\n')
    
        writer.writerow(palavras[:])
        for i, cluster in enumerate(melhor_kmeans):
            for doc in cluster.pontos:
                writer.writerow([i, silhouettes_medios[i], erro])
                writer.writerow(doc.coordenadas)
    
    print("LOG: escrito silhouette + cluster x pontos")
#
##    
##    #Silhouette parte 2
##        
    
    ''' Grafico Silhouette'''  
    #plotting silhouette
    objetos_np = silhouette_y
    y_valores = np.arange(len(objetos_np))
    #print(y_valores)
    #print(type(objetos_np))
    #print(type(y_valores))
    x_valores = silhouette_x
    plt.figure(num=None, figsize=(25, 25), dpi=80)
    barlist = plt.barh(y_valores, x_valores, align='center', alpha=1)
    plt.yticks(y_valores, objetos_np)
    plt.xlabel('Score')
    plt.title('Silhouette para k = ' + str(num_clusters))
    cores = 'rgbkymcrgbkymcrgbkymcrgbkymcrgbkymcrgbkymcrgbkymcrgbkymcrgbkymc'

    #Silhouette por unidade
    for i in range(len(silhouette_cluster)):    
        barlist[i].set_color(cores[silhouette_cluster[i]])


#    # Silhouette para medias
#    for i in range(len(melhor_kmeans)):
#        barlist[i].set_color(cores[i])
        
    silhouette_name = arquivo.replace(".csv", "_") + str(num_clusters) + "_silhouette.png"
    plt.savefig(silhouette_name)
    plt.show()

   
    A = np.array(centros)
    
    setosa = np.array(setosa)
    versicolor = np.array(versicolor)
    virginica = np.array(virginica)
   
    plt.scatter(versicolor[:,0],  versicolor[:,1], s = 100, c = 'green', label = 'Iris-versicolour')
    plt.scatter(setosa[:,0], setosa[:,1], s = 100, c = 'red', label = 'Iris-setosa')
    plt.scatter(virginica[:,0], virginica[:,1], s = 100, c = 'blue', label = 'Iris-virginica')
    plt.scatter(A[:,0], A[:,1], s = 300, c = 'yellow',label = 'Centroids')
    
    plt.title('Iris Clusters and Centroids')
    plt.xlabel('SepalLength')
    plt.ylabel('SepalWidth')
    plt.savefig(str(num_clusters)+ "_SLXSW")
    plt.legend()
     
    plt.show()


    plt.scatter(versicolor[:,0],  versicolor[:,2], s = 100, c = 'green', label = 'Iris-versicolour')
    plt.scatter(setosa[:,0], setosa[:,2], s = 100, c = 'red', label = 'Iris-setosa')
    plt.scatter(virginica[:,0], virginica[:,2], s = 100, c = 'blue', label = 'Iris-virginica')
    plt.scatter(A[:,0], A[:,2], s = 300, c = 'yellow',label = 'Centroids')
    
    plt.title('Iris Clusters and Centroids')
    plt.xlabel('SepalLength')
    plt.ylabel('PetalLength')
    plt.savefig(str(num_clusters)+ "_SLXPL")            
    plt.legend()
     
    plt.show()


    plt.scatter(versicolor[:,0],  versicolor[:,3], s = 100, c = 'green', label = 'Iris-versicolour')
    plt.scatter(setosa[:,0], setosa[:,3], s = 100, c = 'red', label = 'Iris-setosa')
    plt.scatter(virginica[:,0], virginica[:,3], s = 100, c = 'blue', label = 'Iris-virginica')
    plt.scatter(A[:,0], A[:,3], s = 300, c = 'yellow',label = 'Centroids')
    
    plt.title('Iris Clusters and Centroids')
    plt.xlabel('SepalLength')
    plt.ylabel('PetalWidth')
    plt.savefig(str(num_clusters)+ "_SLXPW")            
    plt.legend()
     
    plt.show()


    plt.scatter(versicolor[:,1],  versicolor[:,2], s = 100, c = 'green', label = 'Iris-versicolour')
    plt.scatter(setosa[:,1], setosa[:,2], s = 100, c = 'red', label = 'Iris-setosa')
    plt.scatter(virginica[:,1], virginica[:,2], s = 100, c = 'blue', label = 'Iris-virginica')
    plt.scatter(A[:,1], A[:,2], s = 300, c = 'yellow',label = 'Centroids')
    
    plt.title('Vizualição duas váriaveis - Kmeans IRIS')
    plt.xlabel('SepalWidth')
    plt.ylabel('PetalLength')
    plt.savefig(str(num_clusters)+ "_SWXPL")            
    plt.legend()
    plt.show()

    plt.scatter(versicolor[:,1],  versicolor[:,3], s = 100, c = 'green', label = 'Iris-versicolour')
    plt.scatter(setosa[:,1], setosa[:,3], s = 100, c = 'red', label = 'Iris-setosa')
    plt.scatter(virginica[:,1], virginica[:,3], s = 100, c = 'blue', label = 'Iris-virginica')
    plt.scatter(A[:,1], A[:,3], s = 300, c = 'yellow',label = 'Centroids')
    
    plt.title('Vizualição duas váriaveis - Kmeans IRIS')
    plt.xlabel('SepalWidth')
    plt.ylabel('PetalWidth')
    plt.savefig(str(num_clusters)+ "_SWXPW")            
    plt.legend()
     
    plt.show()

    plt.scatter(versicolor[:,2],  versicolor[:,3], s = 100, c = 'green', label = 'Iris-versicolour')
    plt.scatter(setosa[:,2], setosa[:,3], s = 100, c = 'red', label = 'Iris-setosa')
    plt.scatter(virginica[:,2], virginica[:,3], s = 100, c = 'blue', label = 'Iris-virginica')
    plt.scatter(A[:,2], A[:,3], s = 300, c = 'yellow',label = 'Centroids')
    
    plt.title('Vizualição duas váriaveis - Kmeans IRIS')
    plt.xlabel('PetalLength')
    plt.ylabel('PetalWidth')
    plt.savefig(str(num_clusters)+ "_PlXPW")            
    plt.legend()
     
    plt.show()

    
    return erro
    
            
def kmeans_simples(num_clusters, pontos, tipo_dist, cutoff, num_iteracoes):
    """
    # @pontos: e' o conjunto de cada vetor do documento (uma linha)
    # @num_clusters: e' o numero de clusters
    # @cutoff: Valor para indicar que o algoritmo convergiu para uma resposta
    # @num_iteracoes: Numero de iteracoes 
    # @tipo_dist: Tipo de distancia utilizada
    """
    
    
    erros = []
    todos_clusters = []
    for _ in range(num_iteracoes):
        clusters = kmeans(num_clusters, pontos, tipo_dist, cutoff)
        erro = calcularErroKmeans(clusters)
        todos_clusters.append(clusters)
        erros.append(erro)
        #print('colocou o erro')
        

    menor_erro =  min(erros)
    menor_erro_index = erros.index(menor_erro)
    melhor_kmeans = todos_clusters[menor_erro_index]

    return melhor_kmeans, menor_erro

def kmeans(num_clusters, pontos, tipo_dist, cutoff):

    ''' Para kmeans tradicional '''
    #encontrando maior das frequencias
    freq_maximo = 0
    for ponto in pontos:
        temp = max(ponto.coordenadas)
        if (temp > freq_maximo):
            freq_maximo = temp
            #print(freq_maximo)
            
    centroides = [0.0] * num_clusters
    #Pega centroides aleatorios iniciais
    for a in range(num_clusters):
       centroides[a] = pontoAleatorio(0, freq_maximo, len(pontos[0].coordenadas)) 
   
    '''------------------------FIM KMEANS TRADICIONAL'''
    
    
    ''' Para kmeans ++ '''
    
    # Pega centroides iniciais em nossos dados
    #centroides = random.sample(pontos, num_clusters)
    
    '''------------------------FIM KMEANS ++'''

    # Criando os Clusters 
    clusters = [Cluster([elem]) for elem in centroides]

    
    iteracoes_internas = 0
    limite_convergir = 50
    #apenas para quando chegar no break da convergencia no fim do while
    # ou caso passar do limite para convergir
    while True:
        # criar lista de listas para os pontos de cada cluster
        lists = [[] for _ in clusters]
        k_clusters = len(clusters)
        #print( iteracoes_internas )
        iteracoes_internas += 1
        for p in pontos:
            # distancia ponto e centroide inicial 0
            menor_distancia = getDistancia(p, clusters[0].centroid, tipo_dist)
            # define o index para o inicial tambem
            index_cluster = 0

            for i in range(0, k_clusters):
                # Distancia do ponto para o centroide
                distance = getDistancia(p, clusters[i].centroid, tipo_dist)
                # se for menor, entao registra ela como menor
                if distance < menor_distancia:
                    menor_distancia = distance
                    index_cluster = i
            # coloca para o cluster de menor distancia
            lists[index_cluster].append(p)

        # 0 para primeira iteracao
        maior_deslocamento = 0.0

        for i in range(k_clusters):
            # Quao longe cada centroide andou?
            shift = clusters[i].atualizar(lists[i])
            # maior shift para medirmos com o cutoff fornecido
            maior_deslocamento = max(maior_deslocamento, shift)

        # clusters vazios removidos
        clusters = [c for c in clusters if len(c.pontos) != 0]

        # Limite do algoritmo se movimento for menor que cutoff
        if maior_deslocamento < cutoff or iteracoes_internas > limite_convergir:
            #print (iteracoes_internas, ": algoritmo convergiu")
            #print('saiu')
            break
    return clusters


def getSegundoClusterMaisProximo(ponto, cluster, clusters):
    # inicializando com valor conhecido para conseguirmos sobrescrever
    # na prima iteracao a menor distancia
    menor_distancia = 666666.6
    second_closest = cluster
    for c in clusters:
        if (c != cluster):
            # sobrescrevendo na primeira iteracao pois eh nosso valor conhecido
            if (menor_distancia == 666666.6):
                menor_distancia = getDistancia(ponto, c.centroid, 'euclidiana')
                second_closest = c
            else:
                #se for menor, entao guarda essa ate encontrar a menor de todas
                if(getDistancia(ponto, c.centroid, 'euclidiana') < menor_distancia):
                    menor_distancia = getDistancia(ponto, c.centroid, 'euclidiana')
                    second_closest = c
    return second_closest

def getDistancia(a ,b , tipo_dist):
    if (tipo_dist == 'cosseno'):
        return getDistanciaSimilaridadeCosseno(a, b)
    elif (tipo_dist == 'euclidiana'):
        return getDistanciaEuclidiana(a, b)
    else:
        print("distancia nao suportada")

def getDistanciaEuclidiana(a, b):
    # retorna a soma quadrada das distancias
    distanciaAcumulada = 0.0
    for i in range(a.n):
        diferenca = pow((a.coordenadas[i]-b.coordenadas[i]), 2)
        distanciaAcumulada += diferenca

    return distanciaAcumulada

def getDistanciaSimilaridadeCosseno(a, b):
    # retorna a distancia cosseno e nao similaridade cosseno que
    # seria (1 - similaridade cosseno
    return spatial.distance.cosine(a.coordenadas, b.coordenadas)

def pontoAleatorio(lower, upper, n):
    # para o kmeans tradicional, retorna um ponto de n dimensoes aleatorio
    p = Ponto([random.uniform(lower, upper) for _ in range(n)], 'centro')
    return p

def calcularErroKmeans(clusters):

    # distancia de cada ponto para cada centro
    
    distAcumulada = 0
    num_pontos = 0
    for cluster in clusters:
        num_pontos += len(cluster.pontos)
        distAcumulada += cluster.getTotalDistance()

    erro = distAcumulada / num_pontos
    return erro

def distanciaMedia(meuponto, pontos):
    dist_acumulada = 0
    num_pontos = len(pontos)
    for p in pontos:
        dist_acumulada += getDistancia(meuponto, p, 'euclidiana')
    dist_media = dist_acumulada / num_pontos
    return dist_media

def getSilhouetteScore(ponto, pontos_meucluster , pontos_segundomaisprox):
    a = distanciaMedia(ponto, pontos_segundomaisprox)
    b = distanciaMedia(ponto, pontos_meucluster)
    score = (a - b) / max(a,b)
    return score


class Ponto(object):

    def __init__(self, coordenadas, documento):
        self.n = len(coordenadas)
        self.coordenadas = coordenadas
        self.documento = documento
        self.silhouette_score = 0
        self.meuCluster = -1


    def __repr__(self):
        return str(self.documento)    


class Cluster(object):

    def __init__(self, pontos):

        self.pontos = pontos

        self.n = pontos[0].n

        # Inicializando centroide
        self.centroid = self.posicionarCentroid()

    def __repr__(self):

        return str(self.pontos)

    def atualizar(self, pontos):
        # calcula o quanto o centroide andou
        old_centroid = self.centroid
        self.pontos = pontos
        if len(self.pontos) == 0:
            return 0
        self.centroid = self.posicionarCentroid()
        shift = getDistancia(old_centroid, self.centroid, "euclidiana")
        return shift

    def posicionarCentroid(self):
        #recalcula a posicao do centroide
        
        numPoints = len(self.pontos)
        coordenadas = [p.coordenadas for p in self.pontos]
        # agrega as coordenadas (todas as dimensoes)
        unzipped = zip(*coordenadas)
        # calcula a coordenada media nova do centroide
        centroid_coordenadas = [math.fsum(dList)/numPoints for dList in unzipped]

        return Ponto(centroid_coordenadas, "centro")

    def getTotalDistance(self):

        soma = 0.0
        for p in self.pontos:
            soma += getDistancia(p, self.centroid, 'euclidiana')

        return soma



if __name__ == "__main__":
    main()

