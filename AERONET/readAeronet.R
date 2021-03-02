library(dplyr)

aeronet<-read.table("/Users/mf/Documents/Mongolia/aeronet/100101_170831_Dalanzadgad.lev20",sep=",", header=TRUE,skip=4,stringsAsFactors=FALSE)

# average between 10am -12pm
time<-strsplit(aeronet$Time.hh.mm.ss.,":")
aeronet$hour<-as.numeric(sapply(time, "[[", 1) )
aeronet$min<-as.numeric(sapply(time,"[[",2))
aeronet$sec<-as.numeric(sapply(time,"[[",3))

date<-strsplit(aeronet$Date.dd.mm.yy.,":")
#aeronet$day<-as.numeric(sapply(date, "[[", 1) )
#aeronet$month<-as.numeric(sapply(date,"[[",2))
aeronet$year<-as.numeric(sapply(date,"[[",3))
aeronet$date<-as.Date(aeronet$Date.dd.mm.yy.,"%d:%m:%Y")

log_slope= log(aeronet$AOT_675)-log(aeronet$AOT_500)


aeronet.hrs<-aeronet[aeronet$hour %in% c(3,4) & aeronet$year>=2011,]
aeronet_match<- aeronet.hrs %>%  group_by(date) %>%  summarise(AOD_500=mean(AOT_500,na.rm=TRUE), 
                                                              AOD_440=mean(AOT_440,rm=TRUE),
                                                                     n=n()) 
ggplot(aeronet_match, aes(x = date, y=AOD_440))+
  geom_point()+
  labs(x = "Date",y="Aeronet AOD 440 nm")

# interpolate 675 nm and 500 nm to 550 nm

# calculate slope as delta_y/delta_x where x is wavelengths and y is AERONET AOD
# delta_x is log(675)-log(500)
# delta_y is the log AOD difference at these two wavelengths

delta_x=log(675)-log(500)
aeronet$delta_y=log(aeronet$AOT_675)-log(aeronet$AOT_500)
aeronet$slope=aeronet$delta_y/delta_x

delta_x_new=log(550)-log(500)
aeronet$slope_new=aeronet$slope*delta_x_new+log(aeronet$AOT_500)

aeronet$AOT_550=exp(aeronet$slope_new)
summary(AOT_550)


