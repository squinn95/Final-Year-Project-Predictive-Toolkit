#d2=read.table("C:/Users/I320226/Documents/College/FYP/prediction/Iris.csv",sep=";",header=TRUE)
tab <- read.csv(file="C:/Users/I320226/Documents/College/FYP/prediction/Iris.csv", header=TRUE)

tab$Species <- as.character(tab$Species)
tab$Species[tab$Species == "Iris-setosa"] <- 1
tab$Species[tab$Species == "Iris-versicolor"] <- 2
tab$Species[tab$Species == "Iris-virginica"] <- 3
tab$Species <- as.numeric(tab$Species)

#print((tab))
#junk$nm[junk$nm == "B"] <- "b"
write.csv(tab, file = "C:/Users/I320226/Documents/College/FYP/prediction/IrisNumerical.csv", row.names=FALSE)