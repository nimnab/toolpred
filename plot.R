check.install <- function(package) {
  if(is.element(package, rownames(installed.packages())) == FALSE) {install.packages(package)}
  library(package,character.only = TRUE)
}
check.install("reshape")
check.install("dplyr")
check.install('ggplot2')

res <- read.csv("/home/nnabizad/code/toolpred/res/final/kn_plot.csv")



res$Order <- as.factor(res$Order)



p<- ggplot(res, aes(x=Order, y=Accuracy, fill=Training.set)) +scale_fill_grey()+
  geom_bar(stat="identity", color="black", 
           position=position_dodge()) +
  geom_errorbar(aes(ymin=Accuracy-error*Accuracy, ymax=Accuracy+error*Accuracy), width=.2,
                position=position_dodge(.9)) + theme_bw() + geom_text(aes(label=Accuracy), position=position_dodge(width=0.9), vjust=-0.5,size = 10) + theme(text = element_text(size=30))

p
  
res <- read.csv("/home/nnabizad/code/toolpred/res/final/bon_plot_all.csv")

res$Order <- as.factor(res$Order)

sub = subset(res, Order == " All")
sub$Mean <- as.factor(sub$Mean)
sub$Std <- as.factor(sub$Std)

p1 <- ggplot(data=sub, aes(x=Mean, y=Accuracy, group=Std, color = Std)) +
  geom_line()+
  geom_point()+ geom_text(aes(label=Accuracy),vjust=-1) + theme(text = element_text(size=30))
p1


mtd <- read.csv("/home/nnabizad/code/toolpred/res/final/mtd15.csv")
mtd$Order <- mtd$Order+1
mtd$Iteration <- as.factor(mtd$Iteration)
mtd$Order <- as.factor(mtd$Order)

p1 <- ggplot(data=mtd, aes(x=Order, y=Value, group=Iteration, color = Iteration)) +
  geom_line()+
  geom_point() + theme(text = element_text(size=30))+ labs(x = expression(mu), y= expression(psi(mu)))
p1

mtd <- read.csv("/home/nnabizad/code/toolpred/res/final/mtd_plot.csv")
mtd$Iteration <- as.factor(mtd$Iteration)

p1 <- ggplot(data=mtd, aes(x=Iteration, y=Accuracy, group=Order, color = Order)) +
  geom_line(size = 1.5)+
  geom_point() + theme(text = element_text(size=30))
  p1

lens <- read.csv("/home/nnabizad/code/toolpred/lengthsall.csv")

qplot(lens$length,
      geom="histogram",
      binwidth = 5,  
      main = "Histogram for manual lengths", 
      xlab = "Length",  
      fill=I("gray"), 
      col=I("black"), 
      alpha=I(.2),
      xlim=c(1,62))

ggplot(lens, aes(x=length)) + 
  geom_histogram(bins=20, colour='gray',size=1) + theme(text = element_text(size=30))

res <- read.csv("/home/nnabizad/code/toolpred/res/final/high-orders.csv")



res$Model <- as.factor(res$Model)



p<- ggplot(res, aes(x=Model, y=Accuracy, fill=Training.set)) +scale_fill_grey()+
  geom_bar(stat="identity", color="black", 
           position=position_dodge()) + theme_bw() + geom_text(aes(label=Accuracy), position=position_dodge(width=0.9), vjust=-0.5,size = 10) + theme(text = element_text(size=30))+ theme(axis.text.x = element_text(angle=45, hjust=1))

p
    

