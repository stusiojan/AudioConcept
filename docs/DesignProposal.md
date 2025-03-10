
Cel: Stworzenie systemu, który nie tylko klasyfikuje gatunek muzyczny, ale też wskazuje, które elementy utworu (np. rytm, harmonia, instrumenty) wpłynęły na klasyfikację.

- Design Proposal **16.03.25**
	1. bibliografia
		- TODO
	2. rozpiska prac per tydzień
		- TODO
        - Highlevel:
            - przygotowanie środowiska, research artykułów, przygotowanie design proposal
            - ODDANIE DESIGN PROPOSAL
            - implementacja SVM dla danych GTZAN albo ISMIR2004, implementacja audioLIME, testy
            - ODDANIE PROTOTYPU
            - implementacja CNN (zmodyfikowanego wg artykułu), dalsza implementacja post HOC wyjaśnień (np. SHAP)
            - generowanie odsłuchiwalnych fragmentów odpowiedzialnych za klasyfikację
            - ODDANIE KOŃCOWEGO PROJEKTU
	3. planowany zakres eksperymentów
		- TODO
	4. planowana funkcjonalność programu
		- Wykorzystanie prostego klasyfikatora CNN lub Transformer 
		- Implementacja technik XAI (np. LIME lub SHAP) do generowania wyjaśnień
		- Generowanie odsłuchiwalnych fragmentów odpowiedzialnych za klasyfikację (?)
	5. planowany stack technologiczny
		- Kontrola wersji: github
		- Librosa
		- Python
		- audioLIME
		- SHAP
		- TODO

Znaleźliśmy [artykuł](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10605044), który właściwie ktoś już:
    - Podał Datasety do naszego problemu i sposób ekstrakcji cech
    - Zaproponował architekturę SVM i CNN
    - Preprocessing danych (Melspectrogram + FFT)
    - Metryki ewaluacji (acc, F1, precision - standardowe w sumie)

Po naszej stronie zostałaby implementacja tego artykułu oraz dodanie XAI do wyjaśnienia klasyfikacji.
Nie mamy konkretnego planu, jak i w jakiej formie dostarczyć wyjaśnienia.
Czy uda nam się przygotować wycięte fragmenty audio, które najlepiej ilustrują podjętą decyzję,
czy fragmenty Mel-spektrogramu.

Planem minimum projektu byłaby implementacja artykułu i dodanie wyjaśnień korzystając z audioLIME lub / i SHAPa.

