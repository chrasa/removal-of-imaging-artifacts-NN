# ToDo's

-   [ ] U_0 in RtmWaveSolver.py in Fortran order speichern, um für die Integral berechnung leichter zugänglich zu sein -> nicht möglich, da zu zeitaufwendig. 
    -   [ ] Alternative prüfen: U_0 row-major speichern und auf dem cluster komplett in memory laden
-   [ ] Dataclass einführen, über welche die Ausführkonfigurationen gesetzt werden können. z.B. single oder double, GPU oder CPU
    - [x] Ausführung auf CPU oder GPU
    - [x]Single oder Double
    - [ ]Memmap oder memory
-   [ ] rtm_integral.py und RTM.py zusammenführen und RTM on-the-fly berechnen, da es vermutlich schneller ist. On-the-fly berechnung optional machen
-   [ ] plot Skripte in seperaten Ordner packen für mehr Übersicht
-   [ ] *.npy Dateien in seperaten Ordner packen für mehr Übersicht
