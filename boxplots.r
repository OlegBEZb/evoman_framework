library(ggstatsplot)
library(ggplot2)
library(car)

enemies = c("2","3","5")

for (enemy in enemies){
  data <- read.csv(file = paste(paste('plots/boxplotEnemy',enemy,sep = ""),'.csv',sep = ""),colClasses=c("NULL", NA, NA))
  head(data)
  
  equalvariance = leveneTest(data=data, fitness~EA)
  if (isTRUE(equalvariance$`F value`[1]>0.05)){
    isequal = TRUE
  }else{
    isequal = FALSE
  }

  if ((isTRUE(shapiro.test(data$fitness[data$EA=="EA1"])[2]>0.05))&(isTRUE(shapiro.test(data$fitness[data$EA=="EA2"])[2]>0.05))){
    ggbetweenstats(data,x = EA,y = fitness, var.equal = isequal, bf.message = FALSE, xlab = "Evolutionary Algorithm", ylab = "Fitness mean", title = paste("Enemy ",enemy))
  } else{
    ggbetweenstats(data,x = EA,y = fitness, type = "np", bf.message = FALSE, xlab = "Evolutionary Algorithm", ylab = "Fitness mean", title = paste("Enemy ",enemy))
  }
  
  ggsave(paste(paste("plots/BoxplotEnemy",enemy,sep = ""),".png",sep = ""))
}

