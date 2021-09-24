## Krótki przewodnik po kodzie
### nap/monitor.py
Plik z monitorem i monitorem euklidesowym, który dawał słabe wyniki
Funkcja get_comfort_level zwraca najkrótszą odległość Hamminga próbki na wejściu od znanych patternów

### nap/nap.py
Plik z metodą
W funkcji train_H są do wyboru dwa sposoby użycia - find_only_threshold gdy testujemy znalezioną konfigurację; \
find_best_layer_to_monitor do szukania po hipersiatce najlepszych konfiguracji w tym wypadku dla VGG - na końcu wypisuje
wartości skuteczności dla wszystkich konfiguracji

### plot.py
Plik z funkcjami do zapisywania wyników i rysowania wykresów

## Po rezultatach
W plikach .csv są wyniki OD-testu dla wszystkich metod, architektur i datasetów \ 
W pliku cfgs są wybrane przeze mnie konfiguracje dla każdego modelu. Wszystkie poza Resnet_cifar10 monitorują jedną warstwę. \ 
Na przykład l2 a1 q0.5 oznacza że monitoruję drugą warstwę aktywacji po poolingu nn.AdaptiveAvgPool2d(1) i biorę pod uwagę tylko wartości
powyżej 50 centyla. \
Wyniki pośrednie poszukiwań najlepszych konfiguracji dla obu architektur są niżej w folderach, ale na Resnetowe nie ma sensu patrzeć
bo jeszcze wtedy nie zapisywałem tego w uporządkowany sposób.  \
Przykład z vgg/mnist/vsfashion wiersz: [0.1, 0.2, 0.2, 0.95, 0.85, 64, 0, 0.6001] \ 
1 - wartość to centyl/100, tu 10; kolejne 4 można zignorować; 6 - długość monitorowanej warstwy, tu 64; 7 - skuteczność, tu 0.6001 \ 
Dla wszystkich plików z folderu vgg rozpatrywałem takie konfiguracje po siatce - 
dla warstw feature z krokiem 2 lub 3 i obu warstw classifier; dla poolingów od 1 do 3; dla wartości centyli od 10 do 90 z krokiem 20 \
Niestety nie zapisywałem wprost, która to warstwa i który pooling. 
Trzeba to sobie wnioskować wiedząc, że najpierw sprawdza 5 różnych wartości centyli dla warstwy 0. i 1. poolingu, potem 5 dla w. 0. i 2. p. etc. \  





