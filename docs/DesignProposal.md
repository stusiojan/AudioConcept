
## Cel projektu
Stworzenie systemu, który nie tylko klasyfikuje gatunek muzyczny, ale też wskazuje, które elementy utworu (np. rytm, harmonia, instrumenty) wpłynęły na klasyfikację.

Znaleźliśmy [artykuł](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10605044), w którym właściwie ktoś już:
- Podał Datasety do naszego problemu i sposób ekstrakcji cech
- Zaproponował architekturę SVM i CNN
- Preprocessing danych (Melspectrogram + FFT)
- Metryki ewaluacji (acc, F1, precision - standardowe w sumie)

Po naszej stronie zostałaby implementacja tego artykułu oraz dodanie XAI do wyjaśnienia klasyfikacji.
Nie mamy konkretnego planu, jak i w jakiej formie dostarczyć wyjaśnienia.

Nie wiemy czy uda nam się przygotować wycięte fragmenty audio, które najlepiej ilustrują podjętą decyzję lub biny Mel-spektrogramu.

Planem minimum projektu byłaby implementacja artykułu (pomijając metody, które nie były wyrózniające się) i dodanie wyjaśnień korzystając z audioLIME lub / i SHAPa.

## Design Proposal **16.03.25**

### Bibliografia

| Nazwa         | Rozwiązania / Uwagi         | Link |
| ------------- | ------------- |------|
| “Musical Genre Classification Using Advanced Audio Analysis and Deep Learning Techniques”   | Trening klasyfikacji przeprowadzono na zbiorach danych GTZAN oraz ISMIR2004. Do klasyfikacji gatunków muzycznych wykorzystano modele FNN, CNN, RNN-LSTM, SVM i KNN. Preprocessing obejmował ekstrakcję cech (MFCC, FFT, STFT), a optymalizację przeprowadzono za pomocą dropoutu, L2 regularization i batch normalization.| [link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10605044) |
| “Why Should I Trust You?”: Explaining the Predictions of Any Classifier  | LIME library  | [link](https://arxiv.org/pdf/1602.04938) |
| TOWARDS MUSICALLY MEANINGFUL EXPLANATIONS: USING SOURCE SEPARATION  | audioLIME library  | [link](https://arxiv.org/pdf/2009.02051) |
| Constructing adversarial examples to investigate the plausibility of explanations in deep audio and image classifiers  | Researchers tests how plausible the explainers are by feeding them deliberately  perturbed input data. In audio domain LIME was tested and it does not handle it well. "...the explanation method LIME is not able to recover perturbed segments in a satisfactory manner, with even the baseline performing better". The tested audio was recorded voice.  | [link](https://link.springer.com/article/10.1007/s00521-022-07918-7#notes) |
| “audioLIME: Listenable Explanations Using Source Separation” | audioLIME wykorzystuje separację źródeł dźwięku, aby wyjaśnienia były słuchalne. Można stosować do modeli klasyfikujących muzykę aby zrozumieć, które komponenty dźwięku miały kluczowy wpływ na predykcję modelu.| [link](https://arxiv.org/pdf/2008.00582v3.pdf) |
| “Tracing Back Music Emotion Predictions to Sound Sources and Intuitive Perceptual Qualities” | Artykuł rozszerza audioLIME, dodając średniopoziomowe cechy percepcyjne (np. barwę, rytm, dynamikę), aby lepiej zrozumieć, jak model interpretuje emocje w muzyce. Pomaga wykryć bias w modelach klasyfikujących emocje w muzyce i sprawia, że wyjaśnienia są bliższe temu, jak ludzie rozumieją muzykę. | [link](https://arxiv.org/pdf/2106.07787v2.pdf) |
| CNN ARCHITECTURES FOR LARGE-SCALE AUDIO CLASSIFICATION | VGG-ish / Short-chunk CNNs | [link](https://arxiv.org/pdf/1703.09179) |



### Rozpiska prac per tydzień
| **Nr tygodnia** | **Zakres prac** | **Opis działań** |
|---------------|----------------|----------------|
| **1 (17.03–23.03)** | Konfiguracja projektu | Przygotowanie środowiska (instalacja bibliotek, konfiguracja venv) |
| **2 (24.03–30.03)** | Przygotowanie danych | Pobranie i wstępna eksploracja zbiorów GTZAN/ISMIR2004 |
| **3 (31.03–06.04)** | Implementacja klasyfikatora SVM | Implementacja modelu SVM i testy różnych funkcji jądra |
| **4 (07.04–13.04)** | Prototyp + AudioLIME | Implementacja AudioLIME do wyjaśniania klasyfikacji |
| **5 (14.04–20.04)** | Implementacja klasyfikatora short-chunk CNN | Implementacja podstawowej wersji CNN i pierwszy trening + monitorowanie w TensorBoard |
| **6 (21.04–27.04)** | Optymalizacja CNN | Modyfikacja architektury i optymalizacja hiperparametrów + porównanie wyników w W&B + testy |
| **7 (28.04–04.05)** | Przerwa Majówkowa | Doprecyzowanie formy dostarczania wyjaśnień z XAI|
| **8 (05.05–11.05)** | Wprowadzenie SHAP/XAI | Implementacja SHAP do analizy wpływu cech na predykcję + zapis wyników eksperymentów |
| **9 (12.05–18.05)** | Generowanie wyjaśnień audio | Eksperymenty z generowaniem fragmentów audio jako wyjaśnień |
| **10 (19.05–25.05)** | Finalizacja i dokumentacja | Przygotowanie raportu końcowego i prezentacji projektu |


    - Highlevel:
		- przygotowanie środowiska, research artykułów, przygotowanie design proposal
		- ODDANIE DESIGN PROPOSAL
		- implementacja SVM dla danych GTZAN albo ISMIR2004, implementacja audioLIME, testy
		- ODDANIE PROTOTYPU
		- implementacja CNN (zmodyfikowanego wg artykułu), dalsza implementacja post HOC wyjaśnień (np. SHAP)
		- generowanie odsłuchiwalnych fragmentów odpowiedzialnych za klasyfikację
		- ODDANIE KOŃCOWEGO PROJEKTU

### Planowany zakres eksperymentów
- dla modelu CNN:
	- Wpływ architektury sieci – porównanie klasycznej CNN (2 warstwy) z Modified CNN (więcej niż 2 warstwy).
	- Optymalizacja hiperparametrów – testowanie różnych wartości learning rate (0.0001, 0.001) oraz wpływu dropout (0.2, 0.3) i batch normalization.
	- Monitorowanie w TensorBoard / W&B – zapis metryk i wizualizacja przebiegu trenowania
- dla modelu SVM
	- Wpływ parametrów modelu – testowanie różnych funkcji jądra (liniowe, RBF) oraz wartości parametru C.
	- Eksperymenty zapisane w W&B – porównanie wyników różnych parametrów.

### Testy 
- Dokładność, precyzja oraz F1 modelu na różnych zbiorach danych, takich jak GTZAN i ISMIR2004,
- Porównanie wyników klasyfikatora CNN i SVM na tych samych zbiorach (ocena, który model radzi sobie lepiej w kontekście różnych rodzajów muzyki),
- Sprawdzenie, czy wygenerowane wyjaśnienia (fragmenty audio, które wpłynęły na klasyfikację) rzeczywiście odpowiadają elementom utworu, które mają największy wpływ na decyzję modelu
- Testy jednostkowe 
		
### Planowana funkcjonalność programu
- Wykorzystanie prostego klasyfikatora CNN i SVM
- Implementacja technik XAI (LIME lub SHAP) do generowania wyjaśnień
- Generowanie odsłuchiwalnych fragmentów odpowiedzialnych za klasyfikację

### Planowany stack technologiczny
- Kontrola wersji: github
- Logging: Loguru
- Autoformatter: Ruff
- Środowisko wirtualne: venv / conda
- Monitorowanie eksperymentów: TensorBoard / Weights & Biases (W&B)
- Python
	1. ML
	- Skikit-learn
	- pytorch
	2. Analiza dźwięku
	- Librosa
	3. XAI
	- audioLIME
	- SHAP
	4. Przetwarzanie danych
	- NumPy
	- Pandas
	5. Wizualizacja:
	- matplotlib / seaborn

### Wartości numeryczne dla rozpatrywanych gatunków muzycznych:
	blues - 0 | classical - 1 | country - 2 | disco	- 3 | hiphop - 4 | jazz - 5 | metal - 6 | pop - 7 | reggae - 8 | rock - 9

### Składowe, które weszły do modeli:


| Nazwa zmiennej	| Pełna nazwa po polsku |
|-------------------|-----------------------|
length	| Długość nagrania (w sekundach)
chroma_stft_mean | Średnia wartość chromagramu (z przekształcenia STFT)
chroma_stft_var	| Wariancja chromagramu (STFT)
rms_mean | Średnia wartość RMS (energia/głośność sygnału)
rms_var	| Wariancja RMS (zmienność głośności)
spectral_centroid_var | Wariancja środka spektralnego (jasność barwy dźwięku)
spectral_bandwidth_var	| Wariancja szerokości pasma spektralnego (zakres częstotliwości)
rolloff_mean | Średnia wartość częstotliwości roll-off (próg zawierający 85% energii sygnału)
rolloff_var |	Wariancja częstotliwości roll-off
zero_crossing_rate_mean	| Średni współczynnik przejść przez zero (szorstkość/brzmienie)
zero_crossing_rate_var	| Wariancja współczynnika przejść przez zero
harmony_mean | Średnia wartość komponentu harmonicznego (związana z melodią, tonacją)
harmony_var	| Wariancja komponentu harmonicznego
perceptr_mean | Średnia wartość komponentu perkusyjnego (rytmicznego)
perceptr_var | Wariancja komponentu perkusyjnego
tempo | Tempo utworu w BPM (uderzenia na minutę)
mfcc{1-20}_mean | Średnia wartość MFCC
mfcc{1-20}_var | Wariancja MFCC

