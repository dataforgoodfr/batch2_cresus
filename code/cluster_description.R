library(readr)
library(FactoMineR)

export_to_R <- read_csv("~/Documents/1_ECP/3A/0_Data_for_good/batch2_cresus/data/export_to_R.csv")
export_to_R <- export_to_R[, 2:length(export_to_R)]
export_to_R$cluster <-  factor(export_to_R$cluster)
export_to_R <- data.frame(export_to_R)
c <- catdes(donnee = export_to_R, num.var = 70)
table(export_to_R$cluster)