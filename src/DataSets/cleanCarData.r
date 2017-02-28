set <- read.csv(file="C:/Users/I320226/Documents/College/FYP/prediction/cardata.csv", header=FALSE)

colnames(set) <- c("buying","maint","doors","persons","lug_boot","safety","classValue")

set$buying <- as.character(set$buying)   #vhigh, high, med, low. = 1,2,3,4
set$buying[set$buying == "vhigh"] <- 1
set$buying[set$buying == "high"] <- 2
set$buying[set$buying == "med"] <- 3
set$buying[set$buying == "low"] <- 4
set$buying <- as.numeric(set$buying)

set$maint <- as.character(set$maint)   #vhigh, high, med, low. = 1,2,3,4
set$maint[set$maint == "vhigh"] <- 1
set$maint[set$maint == "high"] <- 2
set$maint[set$maint == "med"] <- 3
set$maint[set$maint == "low"] <- 4
set$maint <- as.numeric(set$maint)

set$doors <- as.character(set$doors)   #2, 3, 4, 5more. = 2,3,4,5
set$doors[set$doors == "5more"] <- 5
set$doors <- as.numeric(set$doors)

set$persons <- as.character(set$persons)   #2, 4, more. = 2,4,5
set$persons[set$persons == "more"] <- 5
set$persons <- as.numeric(set$persons)

set$lug_boot <- as.character(set$lug_boot)   #small, med, big. = 1,2,3
set$lug_boot[set$lug_boot == "small"] <- 1
set$lug_boot[set$lug_boot == "med"] <- 2
set$lug_boot[set$lug_boot == "big"] <- 3
set$lug_boot <- as.numeric(set$lug_boot)

set$safety <- as.character(set$safety)   #low, med, high. = 1,2,3
set$safety[set$safety == "low"] <- 1
set$safety[set$safety == "med"] <- 2
set$safety[set$safety == "high"] <- 3
set$safety <- as.numeric(set$safety)

set$classValue <- as.character(set$classValue)   #unacc, acc, good, vgood = 1,2,3,4
set$classValue[set$classValue == "unacc"] <- 1
set$classValue[set$classValue == "acc"] <- 2
set$classValue[set$classValue == "good"] <- 3
set$classValue[set$classValue == "vgood"] <- 4
set$classValue <- as.numeric(set$classValue)

write.csv(set, file = "C:/Users/I320226/Documents/College/FYP/prediction/CarDataNumerical.csv", row.names=FALSE)
#print(set)



