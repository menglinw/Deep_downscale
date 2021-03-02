pkgs = c('data.table', 'leaflet', 'lubridate', 'sf', 'tidyverse')
for(p in pkgs) require(p, character.only = TRUE)
rm(p, pkgs)

data.dir = paste0(getwd(), '/data/')
code.dir = paste0(getwd(), '/code/')
gfx.dir = paste0(getwd(), '/graphics/')
source(paste0(code.dir, 'helper-misc-functions.R'))

kwt = st_read(paste0(data.dir, 'shapes/KWT_adm0.shp'),
              stringsAsFactors = F, quiet = T) %>%
  select(ISO)

kwt.aeronet.files = list.files(
  path = paste0(data.dir,
                '/aeronet/AOD_Level20_All_Points_V3/AOD/AOD20/ALL_POINTS/'),
  pattern = 'Kuwait', full.names = TRUE)

kwt.aeronet.data = list()
for(i in 1:length(kwt.aeronet.files)){
  kwt.aeronet.data[[i]] = 
    read.table(file = kwt.aeronet.files[i],
               sep = ",", header = TRUE,
               skip = 6, stringsAsFactors = FALSE) %>%
    data.table() %>% 
    mutate(date = as.Date(Date.dd.mm.yyyy., "%d:%m:%Y"),
           date_time = ymd_hm(paste(date, 
                                    substr(Time.hh.mm.ss., 1, 5))))
}

kwt.aeronet.dates = kwt.aeronet.data %>% rbindlist() %>% 
  pull(date) %>% unique() %>% 
  as.Date() %>% sort()

kwt.aeronet.sites = kwt.aeronet.data %>% rbindlist() %>% 
  select(site = AERONET_Site_Name, 
         lon = Site_Longitude.Degrees.,
         lat = Site_Latitude.Degrees.) %>% 
  unique()  %>% 
  st_as_sf(coords = c('lon','lat'), crs = 4326, remove = FALSE)

# # AERONET vs. MAIAC -------------------------------------------------------
# kwt.aeronet.maiac.terra =
#   readRDS(paste0(data.dir, 'maiac/kwt-maiac-aeronet-terra-2006-2015.rds'))
# kwt.aeronet.misr = 
#   readRDS(paste0(data.dir, 'misr/kwt-misr-aeronet-2008-2010.rds'))
# theme_set(theme_bw())
# 
# 
# kwt.misr.lm = lm(AOD_550nm ~ aod, data = kwt.aeronet.misr)
# kwt.maiac.lm = lm(AOD_550nm ~ aod_55, data = kwt.aeronet.maiac.terra)
# 
# kwt.aeronet.aod.assess = data.table(
#   instrument = c('MISR', 'MAIAC'),
#   n = c(nrow(kwt.aeronet.misr), nrow(kwt.aeronet.maiac.terra)),
#   r = round(
#     c((kwt.aeronet.misr %>% 
#          select(aod, AOD_550nm) %>% cor())[1,2],
#       (kwt.aeronet.maiac.terra %>% 
#          select(aod_55, AOD_550nm) %>% cor())[1,2]),
#     digits = 3),
#   rmse = round(
#     c(sqrt(sum((kwt.aeronet.misr %>% 
#                   mutate(error = aod - AOD_550nm) %>% 
#                   pull(error)) ** 2) / nrow(kwt.aeronet.misr)),
#       sqrt(sum((kwt.aeronet.maiac.terra %>% 
#                   mutate(error = aod_55 - AOD_550nm) %>% 
#                   pull(error)) ** 2) / nrow(kwt.aeronet.maiac.terra))),
#     digits = 3),
#   bias = round(c(
#     kwt.misr.lm$coefficients[1],
#     kwt.maiac.lm$coefficients[1]), digits = 3),
#   slope = round(c(
#     kwt.misr.lm$coefficients[2],
#     kwt.maiac.lm$coefficients[2]), digits = 3))
# 
# kwt.aeronet.misr %>% 
#   ggplot(aes(x = aod, y = AOD_550nm)) +
#   geom_abline(slope = 1, linetype = 2, color = 'grey48', size = .5) +
#   geom_point() +
#   # geom_smooth(method = 'lm', color = 'red') +
#   geom_abline(intercept = 
#                 kwt.aeronet.aod.assess[instrument == 'MISR']$bias, 
#               slope =
#                 kwt.aeronet.aod.assess[instrument == 'MISR']$slope, 
#               color = 'red', size = .5) +
#   scale_x_continuous(limits = c(0, 1.25), 
#                      breaks = seq(0, 1.2, by = .2),
#                      expand = c(0.01, 0.01)) +
#   scale_y_continuous(limits = c(0, 1.25), 
#                      breaks = seq(0, 1.2, by = .2),
#                      expand = c(0.01, 0.01)) +
#   labs(x = 'MISR AOD', y = 'AERONET AOD') +
#   ggsave(filename = paste0(gfx.dir, 'kwt-aeronet-vs-misr.png'),
#          height = 4, width = 4, dpi = 450)
#   
# kwt.aeronet.maiac.terra %>% 
#   ggplot(aes(x = aod_55, y = AOD_550nm)) +
#   geom_abline(slope = 1, linetype = 2, color = 'grey48', size = .5) +
#   geom_point() +
#   # geom_smooth(method = 'lm', color = 'red') +
#   geom_abline(intercept = 
#                 kwt.aeronet.aod.assess[instrument == 'MAIAC']$bias, 
#               slope =
#                 kwt.aeronet.aod.assess[instrument == 'MAIAC']$slope, 
#               color = 'red', size = .5) +
#   scale_x_continuous(limits = c(0, 2), 
#                      breaks = seq(0, 2, by = .5),
#                      expand = c(0.01, 0.01)) +
#   scale_y_continuous(limits = c(0, 2), 
#                      breaks = seq(0, 2, by = .5),
#                      expand = c(0.01, 0.01)) +
#   labs(x = 'MAIAC AOD', y = 'AERONET AOD') +
#   ggsave(filename = paste0(gfx.dir, 'kwt-aeronet-vs-maiac.png'), 
#          height = 4, width = 4, dpi = 450)
#
# # Map of AERONET sites ----------------------------------------------------
# require(ggmap)
# register_google(key = 'AIzaSyCiSH7mB-mlCPTYmy-6jrSXuUCZKwlqoew')
# 
# kwt.base = get_map("29.301738, 47.945268", zoom = 12, maptype = 'road') %>% 
#   ggmap_bbox() %>% # custom function from StackOverflow to fix ggmap raster projection
#   ggmap(extent='device') +
#   coord_sf(crs = st_crs(3857)) +
#   theme(legend.text=element_text(size=18),
#         legend.title=element_text(size=18),
#         panel.background = element_blank(),
#         axis.line=element_blank(),
#         axis.text=element_blank(),
#         axis.ticks=element_blank(),
#         axis.title=element_blank())
# 
# kwt.base +
#   geom_sf(data = kwt.aeronet.sites %>% st_transform(crs = 3857), 
#           color = 'gray24', fill = 'firebrick1', 
#           shape = 21, size = 4.5, stroke = 1, inherit.aes = FALSE) +
#   ggsave(filename = paste0(gfx.dir, 'kwt-aeronet-sites.jpeg'),
#          height = 6, width = 6, dpi = 600)
# 
# data.table(date = kwt.aeronet.dates) %>% 
#   group_by(year = as.character(year(date))) %>% 
#   summarize(days = n()) %>% 
#   ggplot() + geom_bar(aes(x = year, weight = days), fill = 'dodgerblue3') +
#   labs(x = 'Year', y = 'Days') +
#   ggsave(filename = paste0(gfx.dir, 'kwt-aeronet-days.jpeg'),
#          height = 4, width = 4, dpi = 600)

# time = strsplit(aeronet$Time.hh.mm.ss., ":")
# date = strsplit(aeronet$Date.dd.mm.yyyy., ":")
# aeronet = aeronet %>% 
#   mutate(hour = as.numeric(sapply(time, "[[", 1)),
#          min = as.numeric(sapply(time, "[[", 2)),
#          sec = as.numeric(sapply(time, "[[", 3)),
#          year = as.numeric(sapply(date, "[[", 3)),
#          date = as.Date(Date.dd.mm.yyyy., "%d:%m:%Y"))

# sample codes

# aeronet<-read.table("/Users/mf/Documents/Mongolia/aeronet/100101_170831_Dalanzadgad.lev20",sep=",", header=TRUE,skip=4,stringsAsFactors=FALSE)
# 
# # average between 10am -12pm
# time<-strsplit(aeronet$Time.hh.mm.ss.,":")
# aeronet$hour<-as.numeric(sapply(time, "[[", 1) )
# aeronet$min<-as.numeric(sapply(time,"[[",2))
# aeronet$sec<-as.numeric(sapply(time,"[[",3))
# 
# date<-strsplit(aeronet$Date.dd.mm.yy.,":")
#aeronet$day<-as.numeric(sapply(date, "[[", 1) )
#aeronet$month<-as.numeric(sapply(date,"[[",2))
# aeronet$year<-as.numeric(sapply(date,"[[",3))
# aeronet$date<-as.Date(aeronet$Date.dd.mm.yy.,"%d:%m:%Y")
# 
# log_slope= log(aeronet$AOT_675)-log(aeronet$AOT_500)
# 
# 
# aeronet.hrs<-aeronet[aeronet$hour %in% c(3,4) & aeronet$year>=2011,]
# aeronet_match<- aeronet.hrs %>%  group_by(date) %>%  summarise(AOD_500=mean(AOT_500,na.rm=TRUE), 
#                                                                AOD_440=mean(AOT_440,rm=TRUE),
#                                                                n=n()) 
# ggplot(aeronet_match, aes(x = date, y=AOD_440))+
#   geom_point()+
#   labs(x = "Date",y="Aeronet AOD 440 nm")
# 
# # interpolate 675 nm and 500 nm to 550 nm
# 
# # calculate slope as delta_y/delta_x where x is wavelengths and y is AERONET AOD
# # delta_x is log(675)-log(500)
# # delta_y is the log AOD difference at these two wavelengths
# 
# delta_x=log(675)-log(500)
# aeronet$delta_y=log(aeronet$AOT_675)-log(aeronet$AOT_500)
# aeronet$slope=aeronet$delta_y/delta_x
# 
# delta_x_new=log(550)-log(500)
# aeronet$slope_new=aeronet$slope*delta_x_new+log(aeronet$AOT_500)
# 
# aeronet$AOT_550=exp(aeronet$slope_new)
# summary(AOT_550)
#