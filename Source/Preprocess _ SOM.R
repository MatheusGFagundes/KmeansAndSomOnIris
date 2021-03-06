#Pacote manipula��o de Strings
install.packages("stringr")
require('stringr')

#Arquivo para preprocessar
arq <- file("E:/Documentos/USP 5oSEM/IA2/iris.txt", "r+")
iris <- readLines(con = arq)

print(length(iris))

#colocando os valores em uma lista
t = 1
tam <- length(iris)
cls <- list()
tudo <- matrix(nrow = tam, ncol = 4)
while(t <= tam){
  
  tudo[t,1] <- as.double(str_sub(iris[t], end = 3))
  tudo[t,2] <- as.double(str_sub(iris[t], 5, 7))
  tudo[t,3] <- as.double(str_sub(iris[t], 9, 11))
  tudo[t,4] <- as.double(str_sub(iris[t], 13, 15))
  cls[t] <- str_sub(iris[t], start = 17)
  t = t + 1
}

colnames(tudo) = c("sepal_l", "sepal_w", "petal_l", "petal_w")

tudo <- `rownames<-`(tudo, cls)
summary(tudo)
novo <- scale(tudo)
summary(novo)

write.csv(novo, file = "E:/Documentos/USP 5oSEM/IA2/novo.csv")

#       #
##     ##
###SOM###
##     ##
#       #

install.packages('kohonen')
require('kohonen')
library(kohonen)

install.packages('devtools')
require('devtools')
#codigo usado � o som 


#parametrizar o algoritmo do som 

a <- 7
b <- 7
n <- 21500
x <- 0.08
y <- 0.04

#Self-organising maps (SOM) para mapear espectros ou padr�es de alta dimens�o para 2D; A dist�ncia euclidiana � usada.
play.SOM <- som(scale(tudo), grid = somgrid(a,b, "hexagonal"),alpha = c(x, y), rlen = n)


#plotar O erro de quantiza��o
mean(play.SOM$distances)


#plotar o som, serapando por g�neros

genre <- c("Iris-setosa", "Iris-versicolor", "Iris-virginica")

plot(play.SOM,
     main = "Results",
     type = "mapping",
     shape = "straight",
     col = c(1:5)[as.factor(row.names(tudo))],
     bgcol = "lightgray",
     labels = "x")
legend("topright", inset = .0, title = "Genres",legend= genre,
       fill = c(1:5), horiz=FALSE)


#plotar a quantidade n�mero de objetos mapeados para as unidades individuais.
plot(play.SOM, type="count", main="Node Counts", shape = "straight")

som.hc <- cutree(hclust(object.distances(play.SOM, "codes")), 3)
add.cluster.boundaries(play.SOM, som.hc)

#plotar a soma das dist�ncias para todos os vizinhos imediatos, tamb�m � conhecido como um gr�fico de matriz U
plot(play.SOM, type="dist.neighbours", main = "SOM neighbour distances", shape = "straight")

som.hc <- cutree(hclust(object.distances(play.SOM, "codes")), 3)
add.cluster.boundaries(play.SOM, som.hc)

#plotar a dist�ncia m�dia para o vetor codebook mais pr�ximo durante o treinamento
plot(play.SOM, type="change")

summary(play.SOM)
