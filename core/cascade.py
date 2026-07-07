"""
core/cascade.py — Universal cascade interface (V1 proof).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from config import (
    CLASSIFIER_PATH,
    DEVICE,
    DOA_COVRECON_PATH,
    DOA_HIERGRU_PATH,
    DOA_LISTA_PATH,
    DOA_SIAMESE_PATH,
    N_ANT,
    N_GRID,
    OUTPUT_DIR,
    PROJECT_ROOT,
)
from core.classifier import ComplexNet, canonicalize_counter_uav_snapshot, to_complex_tensor
from core.doa import (
    CovReconstructor,
    DeepLISTA_DOA,
    HierarchicalGRU_DOA,
    SiameseDOAModel,
    build_steering_matrix,
    cov_to_flat,
    flat_to_hermitian,
    root_music,
)
from core.preprocessing import IdentityPreprocessor, Preprocessor


@dataclass
class DomainSpec:
    name: str
    n_ant: int
    array_pos: np.ndarray
    wavelength: float
    n_grid: int
    classes: list[str]
    doa_classes: list[str]
    output_dir: str = OUTPUT_DIR
    classifier_path: str | None = None
    doa_paths: dict[str, str] | None = None
    preprocessing: str = "wiener_rms"

    @classmethod
    def automotive(cls) -> "DomainSpec":
        return cls(
            name="automotive",
            n_ant=8,
            array_pos=np.linspace(0.0, 3.5, 8),
            wavelength=0.0039,
            n_grid=361,
            classes=["direct", "multipath", "two_targets"],
            doa_classes=["direct", "two_targets"],
            output_dir=OUTPUT_DIR,
            classifier_path=CLASSIFIER_PATH,
            doa_paths={
                "siamese": DOA_SIAMESE_PATH,
                "lista": DOA_LISTA_PATH,
                "covrecon": DOA_COVRECON_PATH,
                "hiergru": DOA_HIERGRU_PATH,
            },
            preprocessing="wiener_rms",
        )

    @classmethod
    def counter_uav(cls) -> "DomainSpec":
        from domains.counter_uav.config import DOA_GRID_DEG

        out = str(Path(PROJECT_ROOT) / "output_counter_uav")
        return cls(
            name="counter_uav",
            n_ant=8,
            array_pos=np.linspace(0.0, 3.5, 8),
            wavelength=0.024,
            n_grid=len(DOA_GRID_DEG),
            classes=["clutter", "drone", "bird", "multi"],
            doa_classes=["drone", "bird", "multi"],
            output_dir=out,
            classifier_path=str(Path(out) / "classifier.pth"),
            doa_paths={
                "siamese": str(Path(out) / "doa_siamese.pth"),
                "lista": str(Path(out) / "doa_lista.pth"),
                "covrecon": str(Path(out) / "doa_covrecon.pth"),
                "hiergru": str(Path(out) / "doa_hiergru.pth"),
            },
            preprocessing="raw",
        )


@dataclass
class CascadeResult:
    scene_class: int
    scene_label: str
    scene_logits: np.ndarray
    doa_angles: np.ndarray | None = None
    doa_method: str | None = None
    doa_confidence: np.ndarray | None = None
    doa_spectra: dict[str, np.ndarray] | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class Cascade:
    """signal → preprocess → classify → route → DOA."""

    GRID_DEG = np.linspace(-90, 90, N_GRID)

    def __init__(
        self,
        domain: DomainSpec,
        classifier: Any | None = None,
        doa_models: dict[str, Any] | None = None,
        preprocessor: Preprocessor | None = None,
        device: str | None = None,
    ):
        self.domain = domain
        self.device = device or DEVICE
        self.preprocessor = preprocessor or self._default_preprocessor()
        self.grid_deg = self._default_grid_deg()
        self.classifier = classifier or self._load_classifier()
        self.doa_models = doa_models or self._load_doa_models()

    def _default_grid_deg(self) -> np.ndarray:
        if self.domain.name == "counter_uav":
            from domains.counter_uav.config import DOA_GRID_DEG

            return np.asarray(DOA_GRID_DEG, dtype=np.float32)
        return np.linspace(-90, 90, self.domain.n_grid)

    def _default_preprocessor(self):
        if self.domain.preprocessing == "raw":
            return IdentityPreprocessor()
        return Preprocessor()

    def _load_classifier(self):
        path = self.domain.classifier_path
        if not path or not Path(path).exists():
            return None
        model = ComplexNet(
            input_length=self.domain.n_ant,
            n_classes=len(self.domain.classes),
        ).to(self.device)
        model.load_state_dict(
            torch.load(path, map_location=self.device, weights_only=True)
        )
        model.eval()
        return model

    def _load_doa_models(self) -> dict[str, Any]:
        paths = self.domain.doa_paths or {}
        models: dict[str, Any] = {}
        if Path(paths.get("siamese", "")).exists():
            m = SiameseDOAModel(n_grid=self.domain.n_grid, N_ant=self.domain.n_ant).to(self.device)
            m.load_state_dict(torch.load(paths["siamese"], map_location=self.device, weights_only=True))
            m.eval()
            models["siamese"] = m
        if Path(paths.get("lista", "")).exists():
            if self.domain.name == "counter_uav":
                from domains.counter_uav.physics_xband import steering_matrix_xband

                A_np = steering_matrix_xband(self.grid_deg)
            else:
                A_np, _ = build_steering_matrix(self.domain.array_pos, self.domain.n_grid)
            m = DeepLISTA_DOA(A_np, n_layers=10).to(self.device)
            m.load_state_dict(torch.load(paths["lista"], map_location=self.device, weights_only=True))
            m.eval()
            models["lista"] = m
        if Path(paths.get("covrecon", "")).exists():
            m = CovReconstructor(N_ant=self.domain.n_ant).to(self.device)
            m.load_state_dict(torch.load(paths["covrecon"], map_location=self.device, weights_only=True))
            m.eval()
            models["covrecon"] = m
        if Path(paths.get("hiergru", "")).exists():
            m = HierarchicalGRU_DOA(
                K_windows=3,
                n_sources_max=3,
                coarse_grid=181,
                fine_grid=self.domain.n_grid,
            ).to(self.device)
            m.load_state_dict(torch.load(paths["hiergru"], map_location=self.device, weights_only=True))
            m.eval()
            models["hiergru"] = m
        return models

    def _classify(self, signal: np.ndarray) -> tuple[int, np.ndarray]:
        if self.classifier is None:
            return 0, np.zeros(len(self.domain.classes), dtype=np.float32)
        if self.domain.name == "counter_uav":
            signal = canonicalize_counter_uav_snapshot(signal)
        x = to_complex_tensor(signal.reshape(1, -1, 1)).to(self.device)
        with torch.no_grad():
            logits = self.classifier(x).squeeze(0).cpu().numpy()
        return int(np.argmax(logits)), logits.astype(np.float32)

    def _predict_doa(
        self, signal: np.ndarray, method: str
    ) -> tuple[np.ndarray, np.ndarray | None]:
        model = self.doa_models.get(method)
        if model is None:
            return np.array([]), None

        model.eval()
        with torch.no_grad():
            if method == "siamese":
                x = torch.complex(
                    torch.tensor(signal.real, dtype=torch.float32),
                    torch.tensor(signal.imag, dtype=torch.float32),
                ).unsqueeze(0).unsqueeze(0).to(self.device)
                _, doa_logits, _ = model(x)
                spectrum = torch.sigmoid(doa_logits).squeeze().cpu().numpy()
                peak = int(np.argmax(spectrum))
                return np.array([self.grid_deg[peak]]), spectrum

            if method == "lista":
                y = torch.complex(
                    torch.tensor(signal.real, dtype=torch.float32),
                    torch.tensor(signal.imag, dtype=torch.float32),
                ).unsqueeze(0).to(self.device)
                power, _ = model(y)
                spectrum = torch.sigmoid(power).squeeze().cpu().numpy()
                peak = int(np.argmax(spectrum))
                return np.array([self.grid_deg[peak]]), spectrum

            if method == "covrecon":
                R_flat = cov_to_flat(signal.reshape(1, -1), self.domain.n_ant)
                R_t = torch.tensor(R_flat, dtype=torch.float32).to(self.device)
                out = model(R_t)
                R_rec = flat_to_hermitian(out, self.domain.n_ant)
                angles = root_music(R_rec[0].cpu().numpy(), N_sources=2)
                return angles, None

            if method == "hiergru":
                x = torch.complex(
                    torch.tensor(signal.real, dtype=torch.float32),
                    torch.tensor(signal.imag, dtype=torch.float32),
                ).unsqueeze(0).unsqueeze(0).to(self.device)
                doa_logits, _ = model(x)
                spectrum = torch.sigmoid(doa_logits).squeeze().cpu().numpy()
                peak = int(np.argmax(spectrum))
                return np.array([self.grid_deg[peak]]), spectrum

        return np.array([]), None

    def forward(
        self,
        signal: np.ndarray,
        doa_method: str = "siamese",
        run_all_doa: bool = False,
    ) -> CascadeResult:
        t0 = time.perf_counter()
        signal = np.asarray(signal).reshape(-1)
        model_input = self.preprocessor(signal)
        scene_idx, logits = self._classify(model_input)
        scene_label = self.domain.classes[scene_idx]

        doa_angles = None
        doa_confidence = None
        doa_spectra: dict[str, np.ndarray] = {}

        if scene_label in self.domain.doa_classes and self.doa_models:
            if run_all_doa:
                all_angles = []
                for name in self.doa_models:
                    angles, spectrum = self._predict_doa(model_input, name)
                    if spectrum is not None:
                        doa_spectra[name] = spectrum
                    if len(angles) > 0:
                        all_angles.extend(angles.tolist())
                doa_angles = np.array(all_angles) if all_angles else None
                doa_method = "all"
            else:
                doa_angles, doa_confidence = self._predict_doa(model_input, doa_method)
                if doa_confidence is not None:
                    doa_spectra[doa_method] = doa_confidence

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return CascadeResult(
            scene_class=scene_idx,
            scene_label=scene_label,
            scene_logits=logits,
            doa_angles=doa_angles,
            doa_method=doa_method if doa_angles is not None else None,
            doa_confidence=doa_confidence,
            doa_spectra=doa_spectra or None,
            latency_ms=latency_ms,
            metadata={
                "preprocessing": self.domain.preprocessing,
            },
        )
