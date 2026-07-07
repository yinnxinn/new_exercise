from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MediaAsset:
    doc_id: str
    media_type: str
    path: str
    caption: str = ""


@dataclass
class MultimodalRegistry:
    """Lightweight placeholder for image/PDF/table assets linked to medical docs."""

    assets_by_doc: dict[str, list[MediaAsset]] = field(default_factory=dict)

    def add(self, asset: MediaAsset) -> None:
        self.assets_by_doc.setdefault(asset.doc_id, []).append(asset)

    def get(self, doc_id: str) -> list[MediaAsset]:
        return self.assets_by_doc.get(doc_id, [])
