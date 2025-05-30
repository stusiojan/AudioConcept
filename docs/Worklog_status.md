# Status 29.05.25r.
## Rozpiska prac per tydzień
| **Nr tygodnia** | **Zakres prac** | **Opis działań** | **Realizacja** |
|---------------|----------------|----------------|----------------|
| **1 (17.03–23.03)** | Konfiguracja projektu | Przygotowanie środowiska (instalacja bibliotek, konfiguracja venv) | planowo |
| **2 (24.03–30.03)** | Przygotowanie danych | Pobranie i wstępna eksploracja zbiorów GTZAN/ISMIR2004 | planowo |
| **3 (31.03–06.04)** | Implementacja klasyfikatora SVM | Implementacja modelu SVM i testy różnych funkcji jądra | planowo |
| **4 (07.04–13.04)** | Prototyp + AudioLIME | Implementacja AudioLIME do wyjaśniania klasyfikacji | zawirowania w zespole, oczyszczenie kodu SVM |
| **5 (14.04–20.04)** | Implementacja klasyfikatora short-chunk CNN | Implementacja podstawowej wersji CNN i pierwszy trening + monitorowanie w W&B | implementacja 5-cio wartswowego CNN, trenowanie|
| **6 (21.04–27.04)** | Optymalizacja CNN | Modyfikacja architektury i optymalizacja hiperparametrów + porównanie wyników w W&B + testy | implementacja VGGisha, usprawnianie CNN, analiza idei XAI i dostępnych narzędzi |
| **7 (28.04–04.05)** | Przerwa Majówkowa | Doprecyzowanie formy dostarczania wyjaśnień z XAI| Przerwy nie było, usprawnianie CNN i VGGish |
| **8 (05.05–11.05)** | Wprowadzenie SHAP/XAI | Implementacja SHAP do analizy wpływu cech na predykcję + zapis wyników eksperymentów | Usprawnianie CNN i VGGish, rezygnacja z AudioLime na rzecz SHAP  |
| **9 (12.05–18.05)** | Generowanie wyjaśnień audio | Eksperymenty z generowaniem fragmentów audio jako wyjaśnień | Reimplementacja VGG, usprawnienie CNN, implementacja rozwiązań SHAP (dla poszczególnych obserwacji i całego zbioru danych |
| **10 (19.05–25.05)** | Finalizacja i dokumentacja | Przygotowanie raportu końcowego i prezentacji projektu | Usprawnianie CNN i VGGish, dodanie modulu do przewidywania gatunku, dodanie prostego UI z możliwością predykcji gatunku i wizualizacją XAI|
| **11 (26.05–01.06)** | - | - | Dalsze usprawnianie CNN i VGGish, przygotowanie dokumentacji, raporty dla XAI, dopracowanie demo na UI |


## Planowany zakres eksperymentów
> dla modelu CNN:
>	- Wpływ architektury sieci – porównanie klasycznej CNN (2 warstwy) z Modified CNN (więcej niż 2 warstwy).
>	- Optymalizacja hiperparametrów – testowanie różnych wartości learning rate (0.0001, 0.001) oraz wpływu dropout (0.2, 0.3) i batch normalization.
>	- Monitorowanie w TensorBoard / W&B – zapis metryk i wizualizacja przebiegu trenowania

**Realizacja:**
- Porównanie architektur 5 warstwowego CNN oraz 16 warstwowego VGG + dodatkowo 3 warstwowy CNN, ale nie był bezpośrednio porównywany
- Eksperymenty z learning rate, regularyzacją, augmentacją danych, schedulerem itp.- mało formalnie jest to zebrane
- Monitorowane w TensorBoard + plotowanie


> dla modelu SVM
>	- Wpływ parametrów modelu – testowanie różnych funkcji jądra (liniowe, RBF) oraz wartości parametru C.
>	- Eksperymenty zapisane w W&B – porównanie wyników różnych parametrów.

**Realizacja:**
- zgodnie z załozeniami, ale niezgrabne porównania - brak zbiorczej tabeli lub wykresu, bo zapisujemy tylko najlepsze z testowanych parametrów

## Testy 
> - Dokładność, precyzja oraz F1 modelu na różnych zbiorach danych, takich jak GTZAN i ISMIR2004,
> - Porównanie wyników klasyfikatora CNN i SVM na tych samych zbiorach (ocena, który model radzi sobie lepiej w kontekście różnych rodzajów muzyki),
> - Sprawdzenie, czy wygenerowane wyjaśnienia (fragmenty audio, które wpłynęły na klasyfikację) rzeczywiście odpowiadają elementom utworu, które mają największy wpływ na decyzję modelu
> - Testy jednostkowe 
		
**Realizacja:**
- Testowane tylko na GTZANie, zbieramy loss i accuracy
- Zgodnie z załozeniami porównanie accuracy CNN i SVM + dodatkowo próbki VGGisha (trenowanym na 30 sekundowym audio). Plotowanie Confusion Matrix dla kazdego z modeli.
- BRAK
- BRAK

## Planowana funkcjonalność programu
> - Wykorzystanie prostego klasyfikatora CNN i SVM
> - Implementacja technik XAI (LIME lub SHAP) do generowania wyjaśnień
> - Generowanie odsłuchiwalnych fragmentów odpowiedzialnych za klasyfikację

**Realizacja:**
- Spełnione + VGG i bardziej złozony CNN
- Częściowa - notebooki
- BRAK

## Planowany stack technologiczny
Zmiany względem DesignProposalu to ~~przekreślenia~~ dla nieuzytych elementów oraz **pogrubienia** dla dodanych elementów

- Kontrola wersji: github
- Logging: Loguru
- Autoformatter: ~~Ruff~~ **Black**
- Środowisko wirtualne: venv / conda
- Monitorowanie eksperymentów: TensorBoard / Weights & Biases (W&B)
- Python
	1. ML
	- Skikit-learn
	- pytorch
	2. Analiza dźwięku
    - **Soundfile**
	- Librosa
	3. XAI
	- ~~audioLIME~~
	- SHAP
	4. Przetwarzanie danych
	- NumPy
	- Pandas
    - **scipy**
	5. Wizualizacja:
	- matplotlib / seaborn

Jest tez parę mniejszych pakietów jak *typer, tqdm*

## Dodatkowe

- Ręczna implementacja augmentacja danych muzycznych (przez konflikty z bibliotekami oferującymi je out of the box)
- automatykacja eksperymentów przez skrypty
- zdiagnozowanie błędnego pliku w datasecie
- w miarę ładna praca w repo (ponazywane ładnie branche i commity)

## Do zrobienia

- [ ] uspójnienie requirements i environment
- [ ] dodanie opcji wyboru modelu do predykcji (aktualnie działa tylko dla SVM)
- [ ] testy
