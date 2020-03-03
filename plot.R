check.install <- function(package) {
  if(is.element(package, rownames(installed.packages())) == FALSE) {install.packages(package)}
  library(package,character.only = TRUE)
}
check.install("reshape")
check.install("dplyr")
check.install('ggplot2')

res <- read.csv("/home/nnabizad/code/toolpred/res/final/bon_plot_all.csv")


res$Order <- as.factor(res$Order)



p<- ggplot(res, aes(x=Order, y=Accuracy, fill=Supervision)) +scale_fill_grey()+
  geom_bar(stat="identity", color="black", 
           position=position_dodge()) +
  geom_errorbar(aes(ymin=Accuracy-error*Accuracy, ymax=Accuracy+error*Accuracy), width=.2,
                position=position_dodge(.9)) + theme_bw() + geom_text(aes(label=Accuracy), position=position_dodge(width=0.9), vjust=-0.5,size = 5) + theme(text = element_text(size=30))

p
  
res <- read.csv("/home/nnabizad/code/toolpred/res/final/bon_plot_sup.csv")

res$Order <- as.factor(res$Order)

sub = subset(res, Order == " All")
sub$Mean <- as.factor(sub$Mean)
sub$Std <- as.factor(sub$Std)

p1 <- ggplot(data=sub, aes(x=Mean, y=Acuracy, group=Std, color = Std)) +
  geom_line()+
  geom_point()+ geom_text(aes(label=Acuracy),vjust=-1) + theme(text = element_text(size=30))
p1


mtd <- read.csv("/home/nnabizad/code/toolpred/res/final/mtd15.csv")
p1 <- ggplot(data=mtd, aes(x=Order, y=Value, group=Std, color = Std)) +
  geom_line()+
  geom_point()+ geom_text(aes(label=Acuracy),vjust=-1) + theme(text = element_text(size=30))
