
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
| TOWARDS MUSICALLY MEANINGFUL EXPLANATIONS: USING SOURCE SEPARATION}  | audioLIME library  | [link](https://arxiv.org/pdf/2009.02051) |
| Constructing adversarial examples to investigate the plausibility of explanations in deep audio and image classifiers  | Researchers tests how plausible the explainers are by feeding them deliberately  perturbed input data. In audio domain LIME was tested and it does not handle it well. "...the explanation method LIME is not able to recover perturbed segments in a satisfactory manner, with even the baseline performing better". The tested audio was recorded voice.  | [link](https://link.springer.com/article/10.1007/s00521-022-07918-7#notes) |
| {nazwa}  | {rozwiązania}  | {link} |




### Rozpiska prac per tydzień
- TODO
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
- dla modelu SVM
	- Wpływ parametrów modelu – testowanie różnych funkcji jądra (liniowe, RBF) oraz wartości parametru C.
		
### Planowana funkcjonalność programu
- Wykorzystanie prostego klasyfikatora CNN i SVM
- Implementacja technik XAI (np. LIME lub SHAP) do generowania wyjaśnień
- Generowanie odsłuchiwalnych fragmentów odpowiedzialnych za klasyfikację (?)

### Planowany stack technologiczny
- Kontrola wersji: github
- Logging: Loguru
- Autoformatter: Ruff
- Środowisko wirtualne: venv
- Librosa
- Python
- audioLIME
- SHAP



