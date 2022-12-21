library(adabag)
library(e1071)
library(randomForest)
library(caret)
library(MASS)
library(kernlab)

error.rate <- function(dataA, dataB) sum( dataA != dataB ) / length(dataB)

# -------------------------------------------------------------------------------------
#general forward greedy selection function
#input:
# x,y inputs and targets
# method is an external function that estimates classification error with a given model
# ... parameters for method
#output:
#ordered.names.list <- nombre de las variables ordenadas de la mas importante a la menos
#ordered.features.list <-numero de orden inicial de las variables, con el mismo orden
#importance <- importancia de cada variables en el mismo orden

#-------------------------------------------------------------------------------------

forward.ranking <- function(x,y,method,verbosity=0,... ) {

	max.feat<-dim(x)[2] # Número total de variables
	num.feat<-1 # Número de variables elegidas en el momento
	list.feat<-1:max.feat

	# ranking inicial: elijo la variable con menor error de prediccion
  x.train<-matrix(0,dim(x)[1],1) # Variable de entrenamiento con solo las columnas elegidas
	class.error<-double(max.feat) # Error de cada variable
	# para cada i, creo el dataset con esa variable sola, entreno un modelo
	# y le mido el error, que lo guardo en class.error[i]
	for(i in 1:max.feat) {
		x.train[,1]<-x[,i]
		class.error[i] <- do.call(method, c(list(x.train, y), list(...)) )
	}
	# guardo la variable con minimo error como primera. Guardo una lista
	# keep.feat con las que me quedan para seguir eligiendo.
	list.feat[1]<-which.min(class.error)
	keep.feat<-sort(class.error,decreasing=FALSE,index=T)$ix[-1]
	# armo un dataset con las variables que ya elegi, para ir agregando en cada paso.
	x.prev<-x.train[,1]<-x[,list.feat[1]]

	if(verbosity>1) cat("\nFirst chosen feature: ",colnames(x)[list.feat[1]],"\n")

  # loop principal. A cada paso agrego todas las variables disponibles,
	# de a una, le mido el error y me quedo con la de minimo error.
	# Hasta llegar a meter todas.
	while(num.feat < max.feat) {
    # class.error guarda el error de cada modelo. Son max.feat-num.feat modelos.
		class.error<-double(max.feat - num.feat)
		# para cada variable que me queda, la agrego al dataset del paso anterior,
		# entreno el modelo y le mido el error.
		for(i in 1:(max.feat-num.feat)){
			x.train<-cbind(x.prev,x[,keep.feat[i]])
			class.error[i] <- do.call(method, c(list(x.train, y), list(...)) )
		}
		if(verbosity>2) cat("\nFeatures:\n",keep.feat,"\nErrors:\n",class.error)
		# me quedo con el modelo de minimo error, guardo ese feature en la lista
		# de las elegidas, lo saco de la lista de las que quedan, y actualizo
		# el dataset de partida de la iteracion.
		best.index<-which.min(class.error)
		list.feat[num.feat+1]<-keep.feat[best.index]
		if(verbosity>1) cat("\n---------\nStep ",1+num.feat,"\nChosen feature:",colnames(x)[keep.feat[best.index]])

		keep.feat<-keep.feat[-best.index]
		if(verbosity>2) cat("\nNew search list: ",keep.feat)
		num.feat<-num.feat+1
		x.prev<-x[,list.feat[1:num.feat]]
	}


	search.names<-colnames(x)[list.feat]
	# le asigno a cada feature una importacia proporcional al orden en que
	# lo seleccionamos
	imp<-(max.feat:1)/max.feat
	names(imp)<-search.names

	if(verbosity>0) {
		cat("\n---------\nFinal ranking ",num.feat," features.")
		cat("\nFeatures: ",search.names,"\n")
	}

 	return( list(ordered.names.list=search.names,ordered.features.list=list.feat,importance=imp) )
}

#---------------------------------------------------------------------------
#random forest error estimation (OOB) for greedy search
#---------------------------------------------------------------------------
rf.est <- function(x.train,y,equalize.classes=TRUE,tot.trees=500,mtry=0) {
	if(mtry<1) mtry<-floor(sqrt(dim(x.train)[2]))
	prop.samples<-table(y)
	if(equalize.classes) prop.samples<-rep(min(prop.samples),length(prop.samples))
	return( randomForest(x.train,y,mtry=mtry,ntree=tot.trees,sampsize=prop.samples)$err.rate[tot.trees] )
}

#---------------------------------------------------------------------------
#LDA error estimation (LOO) for greedy search
#---------------------------------------------------------------------------
lda.est <- function(x.train,y) {
	m.lda <- lda(x.train,y,CV=TRUE)
	return(error.rate( y , m.lda$class))
}

#---------------------------------------------------------------------------
#SVM error estimation (internal CV) for greedy search
#---------------------------------------------------------------------------
svm.est <- function(x.train,y,type="C-svc",kernel="vanilladot",C=1,cross = 4)
{
	return ( ksvm(x.train, y, type=type,kernel=kernel,C=C,cross = cross)@cross )
}

#---------------------------------------------------------------------------
#random forest ranking method for rfe.
#---------------------------------------------------------------------------
imp.rf <- function(x.train,y,equalize.classes=TRUE,tot.trees=500,mtry=0)
{
	if(mtry<1) mtry<-floor(sqrt(dim(x.train)[2]))
	prop.samples<-table(y)
	if(equalize.classes) prop.samples<-rep(min(prop.samples),length(prop.samples))
	
	m.rf<-randomForest(x.train,y,ntree=tot.trees,mtry=mtry,sampsize=prop.samples,importance=TRUE)
	imp.mat<-importance(m.rf)
	imp.col<-dim(imp.mat)[2]-1
	rank.list<-sort(imp.mat[,imp.col],decreasing=FALSE,index=T)
	return(list(feats=rank.list$ix,imp=rank.list$x))
}

#---------------------------------------------------------------------------
#linear svm ranking method for rfe. Using kernlab. Multiclass
#---------------------------------------------------------------------------
imp.linsvm <- function(x.train,y,C=100)
{
	num.feat<-dim(x.train)[2]
	tot.problems<-nlevels(y)*(nlevels(y)-1)/2

	invisible(capture.output(m.svm <- ksvm(as.matrix(x.train), y, type="C-svc",kernel="vanilladot",C=C)))

	w<-rep(0.0,num.feat)
	for(i in 1:tot.problems) for(feat in 1:num.feat)
		w[feat]<-w[feat]+abs(m.svm@coef[[i]] %*% m.svm@xmatrix[[i]][,feat])
	rank.list<-sort(w,decreasing=FALSE,index=T)
	return(list(feats=rank.list$ix,imp=rank.list$x))
}

error.rate <- function(dataA, dataB) sum( dataA != dataB ) / length(dataB)

#hacer una funcion que cree datos, 2 clases (-1 y 1,n puntos de cada una), d dimensiones, de ruido uniforme [-1,1], con la clase al azar
crea.ruido.unif<-function(n=100,d=2){
	x<-runif(2*n*d,min=-1)	#genero los datos
	dim(x)<-c(2*n,d)
	return(cbind(as.data.frame(x),y=factor(rep(c(-1,1),each=n))))	#le agrego la clase
}

backward.ranking <- function(x,y,method,verbosity=0,... ) {
	max.feat<-dim(x)[2] # Número total de variables
	num.feat <- max.feat
	discarded.feat<-1:max.feat
	keep.feat <- discarded.feat

  x.train<-matrix(0,dim(x)[1],1) # Variable de entrenamiento con solo las columnas elegidas
	x.prev <- x

  # loop principal. A cada paso quito todas las variables disponibles,
	# de a una, le mido el error y descarto con la de mínimo error.
	# hasta no quedarme con ninguna.
	while(num.feat > 1) {
    # class.error guarda el error de cada modelo. Son num.feat modelos.
		class.error<-double(num.feat)
		# para cada variable que me queda, la saco del dataset del paso anterior,
		# entreno el modelo y le mido el error.
		for(i in 1:num.feat){
			x.train<-x.prev[,-i, drop=FALSE]
			class.error[i] <- do.call(method, c(list(x.train, y), list(...)) )
		}
		if(verbosity>2) cat("\nFeatures:\n",keep.feat,"\nErrors:\n",class.error)
		# me quedo con el modelo de minimo error, guardo ese feature en la lista
		# de las descartadas, lo saco de la lista de las que quedan, y actualizo
		# el dataset de partida de la iteracion.
		worst.index<-which.min(class.error)
		discarded.feat[(max.feat - num.feat) + 1]<-keep.feat[worst.index]
		if(verbosity>1) cat("\n---------\nStep ",(max.feat - num.feat) + 1,"\nDiscarded feature: ",colnames(x)[keep.feat[worst.index]])

		keep.feat<-keep.feat[-worst.index]
		if(verbosity>2) cat("\nNew search list: ",keep.feat)
		num.feat<-num.feat-1
		x.prev<-x[,keep.feat[1:num.feat], drop=FALSE] #TODO esta línea
	}

	# la única variable que quedó sin descartar es la más importante
	discarded.feat[max.feat]<-keep.feat[1]
	if (verbosity > 1) cat("\nLast discarded feature: ",colnames(x)[keep.feat[1]])

	# las variables menos importantes se descartaron primero -> las mas importantes
	# están al final, doy vuelta la lista
	list.feat <- rev(discarded.feat)
	search.names<-colnames(x)[list.feat]
	# le asigno a cada feature una importacia proporcional al orden en que
	# lo seleccionamos
	imp<-(max.feat:1)/max.feat
	names(imp)<-search.names

	if(verbosity>0) {
		cat("\n---------\nFinal ranking ",max.feat," features.")
		cat("\nFeatures: ",search.names,"\n")
	}

 	return( list(ordered.names.list=search.names,ordered.features.list=list.feat,importance=imp) )
}

kruskalwallis.ranking <- function(x, y, verbosity=0,...) {
  num.feat <- dim(x)[2]
  class.statistic <- double(num.feat)

  for (i in 1:num.feat){
    x.test <- x[,i]
    class.statistic[i] <- kruskal.test(x.test,y)$statistic
    if (verbosity > 1) {
      cat("\n---------\nFeature ",i,colnames(x)[i],"\nKruskal-Wallis chi-squared:",class.statistic[i])
    }
  }

  list.feat <- sort(class.statistic,decreasing=TRUE,index=T)$ix
  search.names <- colnames(x)[list.feat]

  # le asigno a cada feature una importacia proporcional al orden en que
	# lo seleccionamos
	imp <- (num.feat:1)/num.feat
	names(imp) <- search.names

  if (verbosity > 0) {
		cat("\n---------\nFinal ranking ",num.feat," features.")
		cat("\nFeatures: ",search.names,"\n")
	}

  return( list(ordered.names.list=search.names,ordered.features.list=list.feat,importance=imp) )
}

rfe.ranking <- function(x,y,method,verbosity=0,... ) {
	num.feat <- dim(x)[2]
	discarded.feat <- 1:num.feat
	keep.feat <- discarded.feat

	for (i in 1:num.feat) {
		x.train <- as.matrix(x[,keep.feat])
		rankings <- do.call(method, c(list(x.train, y), list(...)) )
		worst.index <- rankings$feats[1]
		if(verbosity>1) cat("\n---------\nStep ",i,"\nDiscarded feature: ",colnames(x)[keep.feat[worst.index]],"\n")
		discarded.feat[i] <- keep.feat[worst.index]
		keep.feat <- keep.feat[-worst.index]
	}


	# las variables menos importantes se descartaron primero -> las mas importantes
	# están al final, doy vuelta la lista
	list.feat <- rev(discarded.feat)
	search.names<-colnames(x)[list.feat]
	# le asigno a cada feature una importacia proporcional al orden en que
	# lo seleccionamos
	imp<-(num.feat:1)/num.feat
	names(imp)<-search.names

	if(verbosity>0) {
		cat("\n---------\nFinal ranking ",num.feat," features.")
		cat("\nFeatures: ",search.names,"\n")
	}

 	return( list(ordered.names.list=search.names,ordered.features.list=list.feat,importance=imp) )
}

compare.clusters <- function(clusters1, clusters2) {
  cont.table <- table(clusters1, clusters2)
  class.match <- matchClasses(as.matrix(cont.table),method="exact")
  print(cont.table[,class.match])
}

gap.statistic <- function(x, max.k, b.references) {
  # Como el k más chico que busco depende de k + 1 calculo para max.k + 1
  # pero luego devuelvo los primeros max.k (aunque probablemente corte antes de
  # llegar a esto)
  max.k <- max.k + 1
  rows <- nrow(x)
  columns <- ncol(x)

  gaps <- double(max.k)
  s <- double(max.k)

  # Voy a generar los datasets de referencia sobre la PCA de los datos.
  # Calculo los máximos y minimos para cada variable de la PCA para generar
  # con una distribución uniforme sobre esos valores
  x.pca <- prcomp(x)$x
  x.mins <- apply(x.pca, 2, min)
  x.maxs <- apply(x.pca, 2, max)

  # Genero los datasets de referencia, utilizo siempre los mismos b datasets para
  # todos los k para tener resultados mas consistentes
  reference.datasets <- list()
  for (b in 1:b.references) {
    dataset <- c()
    for (i in 1:rows) {
      dataset <- rbind(dataset, runif(columns, min=x.mins, max=x.maxs))
    }
    reference.datasets[[b]] <- dataset
  }

  for (k in 1:max.k) {
    # Calculo la dispersión para este k con los datos originales
    Wk <- kmeans(x, cent=k, nstart=10, iter.max=30)$tot.withinss

    # Calculo la dispersión para los datasets de referencia
    Wkb <- double(b.references)
    for (b in 1:b.references) {
      Wkb[b] <- kmeans(reference.datasets[[b]], cent=k, nstart=10, iter.max=30)$tot.withinss
    }

    # Un poco de trabalengua estadístico
    l <- (1 / b.references) * sum(log(Wkb))
    sdk <- sqrt((1 / b.references) * sum((log(Wkb) - l)^2))
    s[k] <- sdk * sqrt(1 + (1 / b.references))

    gaps[k] <- (1 / b.references) * sum(log(Wkb) - log(Wk))
  }

  # Ahora a buscar el mínimo k tal que gaps[k] >= gaps[k + 1] - s[k + 1]
  best.k <- 2 # Arranco en 2 o se queda casi siempre en 1
  while (best.k < max.k && 
         gaps[best.k] < gaps[best.k + 1] - s[best.k + 1]) {
    best.k <- best.k + 1
  }

  return(list(best.k = best.k, gaps = gaps[1:max.k - 1]))
}

# Función que calcula el score de estabilidad  de dos soluciones de clustering
# (misma del enunciado)
stability.score <- function(n, ind1, cc1, ind2, cc2) {
  #pongo los clusters de nuevo en longitud n - quedan 0 los puntos fuera del sample
  v1<-v2<-rep(0,n)
  v1[ind1]<-cc1+5
  v2[ind2]<-cc2+5
  #creo una matriz m con 1 donde los dos puntos estan en el mismo cluster, -1 en distinto cluster y 0 si alguno no esta, para cada
  # clustering
  a<-sqrt(v1%*%t(v1))
  m1<-a / -a + 2*(a==round(a))
  m1[is.nan(m1)]<-0
  a<-sqrt(v2%*%t(v2))
  m2<-a / -a + 2*(a==round(a))
  m2[is.nan(m2)]<-0
  #calculo el score, los pares de puntos que estan en la misma situacion en los dos clustering dividido el total de pares validos.
  validos<-sum(v1*v2>0)
  sum((m1*m2)[upper.tri(m1)]>0)/(validos*(validos-1)/2)
}

# Recibo los datos que quiero clusterizar, la cantidad maxima de clusters,
# la cantidad de replicas que voy a generar para cada k
# y el porcentaje utilizado para hacer subsampling
stability <- function(x, max.k, replics, sub.percent=0.9) {
  all.scores = list()
  rows <- dim(x)[1]

  # Preparo los indices para subsamplear para cada k (datos perturbados)
  subsamples.ind <- c()
  for (rep in 1:replics) {
    subsamples.ind <- rbind(subsamples.ind, sample(rows,sub.percent*rows))
  }

  for (k in 2:max.k) {
    # Hago kmeans para cada subset de datos perturbados
    kmeans <- c()
    for (rep in 1:replics) {
      kmeans <- rbind(kmeans, kmeans(x[subsamples.ind[rep,],], cent=k,nstart=10, iter.max=30)$cluster)
    }

    # Calculo la estabilidad entre cada par de réplicas para luego calcular la media
    scores <- c()
    for (ind1 in 1:(replics - 1)) {
      for (ind2 in (ind1 + 1):replics) {
        scores <- c(scores, stability.score(rows, subsamples.ind[ind1,], kmeans[ind1,], subsamples.ind[ind2,], kmeans[ind2, ]))
      }
    }
    all.scores[[k]] <- scores
  }

  sorted <- lapply(all.scores, sort)
	cumulative <- lapply(sorted, cumsum)
	
	return(list(sorted=sorted, cumulative=cumulative))
}
