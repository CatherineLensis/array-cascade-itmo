# Cascade architecture diagram

```mermaid
flowchart LR
    signal[Signal N_ant complex] --> preprocess[Preprocess WienerHopf RMS]
    preprocess --> classify[Scene Classifier ComplexNet]
    classify --> router[Router core/cascade]
    router -->|doa_classes| doa[DOA 4 methods]
    router -->|skip| skip[No DOA]
    doa --> result[Angles + confidence]
```

## Two-domain proof

| Component | Automotive | Counter-UAV |
|---|---|---|
| Wavelength | 77 GHz | 12.5 GHz X-band |
| Classes | 3 | 4 |
| Cascade core | shared | shared |
| Physics | domains/automotive | domains/counter_uav |
