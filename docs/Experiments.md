**weight decay**

1e-4, 1e-3, 5e-4, 5e-3 ostatecznie dla dłuzszych batchy jest 8e-4

Wyniki z WEIGHT_DECAY = 1e-4 osiągnęły acc = 0.5 w 9 epoce
![](../reports/figures/VGGish_training_plots_20250528_1833.png)
Wyniki z WEIGHT_DECAY = 1e-3 osiągnęły acc=0.52 w 9 epoce
![](../reports/figures/VGGish_training_plots_20250528_1852.png)
Wyniki z WEIGHT_DECAY = 5e-4 osiągnęły acc=0.57 w 9 epoce
![](../reports/figures/VGGish_training_plots_20250528_2006.png)
Wyniki z WEIGHT_DECAY = 5e-3 osiągnęły acc=0.46 w 8 epoce
![](../reports/figures/VGGish_training_plots_20250528_2028.png)

Najlepsze rezultaty w 9 epoce prawdopodobnie wynikają ze schedulera ReduceLROnPlateau z patience równym 8

Dluzszy trening z 8e-4
![](../reports/figures/VGGish_training_plots_20250528_2255.png)

1x30sek vs 1x3sek vs 10x3sek

Mogę puścić jeszcze raz, ale ogólnie samo accuracy jest podobne

**learning rate**

Tego było duzo, ostatecznie dla CNN najlepsze wyniki otrzymuję przy kroku 7e-3, a VGGish przy 3e-5

**different schedulers for longer lerning sessions*8