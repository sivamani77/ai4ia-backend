from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.db.postgres import pg_fetch_all, pg_fetch_one
from app.services.speaker_names import norm_key

if TYPE_CHECKING:
    from app.services.speaker_registry import SpeakerRegistry

logger = logging.getLogger("ai4ia.db.repositories.speakers")

# Rows written from the registry carry this source. Rows with no source were
# written by the old in-code alias seed. Only these two are ever deleted.
REGISTRY_SOURCE = "registry"


class SpeakerRepository:
    """speakers, speaker_aliases and video_speakers, all derived from the
    speaker registry (config/speaker_registry.json)."""

    def upsert_from_registry(self, registry: "SpeakerRegistry") -> dict[str, int]:
        names = list(registry.speakers)
        if not names:
            return {"speakers": 0, "aliases": 0}
        entries = [registry.speakers[n] for n in names]
        pg_fetch_all(
            """
            insert into speakers (canonical_name, status, source, register_name, affiliation, updated_at)
            select u.name, u.status, %s, u.register_name, u.affiliation, now()
            from unnest(%s::text[], %s::text[], %s::text[], %s::text[])
                 as u(name, status, register_name, affiliation)
            on conflict (canonical_name) do update set
                status = excluded.status,
                source = excluded.source,
                register_name = excluded.register_name,
                affiliation = excluded.affiliation,
                updated_at = now()
            returning canonical_name
            """,
            (REGISTRY_SOURCE, names, [e.get("status") or "confirmed" for e in entries],
             [e.get("register_name") for e in entries], [e.get("affiliation") for e in entries]),
        )
        aliases = registry.aliases()
        if aliases:
            pg_fetch_all(
                """
                insert into speaker_aliases (alias, canonical_name, normalized_alias, source)
                select u.alias, u.canonical_name, u.normalized_alias, %s
                from unnest(%s::text[], %s::text[], %s::text[]) as u(alias, canonical_name, normalized_alias)
                on conflict (alias) do update set
                    canonical_name = excluded.canonical_name,
                    normalized_alias = excluded.normalized_alias,
                    source = excluded.source
                returning alias
                """,
                (REGISTRY_SOURCE, list(aliases), list(aliases.values()), [norm_key(a) for a in aliases]),
            )
        return {"speakers": len(names), "aliases": len(aliases)}

    def delete_not_in_registry(self, registry: "SpeakerRegistry") -> dict[str, int]:
        """Remove registry-owned or legacy rows the registry no longer lists."""
        aliases = pg_fetch_all(
            """
            delete from speaker_aliases
            where (source is null or source = %s) and not (alias = any(%s))
            returning alias
            """,
            (REGISTRY_SOURCE, list(registry.aliases())),
        )
        speakers = pg_fetch_all(
            """
            delete from speakers s
            where (s.source is null or s.source = %s)
              and not (s.canonical_name = any(%s))
              and not exists (select 1 from speaker_aliases a where a.canonical_name = s.canonical_name)
            returning canonical_name
            """,
            (REGISTRY_SOURCE, list(registry.speakers)),
        )
        return {"aliases": len(aliases), "speakers": len(speakers)}

    def replace_video_speakers(self, registry: "SpeakerRegistry") -> int:
        """Make video_speakers match the registry's roster for every registry video."""
        video_ids, names, roles, ordinals = [], [], [], []
        for video_id in registry.video_ids():
            entry = registry.video(video_id)
            for ordinal, (name, role) in enumerate(entry.speakers if entry else ()):
                video_ids.append(video_id)
                names.append(name)
                roles.append(role)
                ordinals.append(ordinal)
        # Two statements, each safe to repeat: a delete and an insert in one
        # statement (a CTE) cannot see each other and collide on the key.
        rows = pg_fetch_all(
            """
            insert into video_speakers (video_id, canonical_name, role, ordinal)
            select * from unnest(%s::text[], %s::text[], %s::text[], %s::int[])
            on conflict (video_id, canonical_name) do update set
                role = excluded.role,
                ordinal = excluded.ordinal
            returning video_id
            """,
            (video_ids, names, roles, ordinals),
        )
        pg_fetch_all(
            """
            delete from video_speakers v
            where v.video_id = any(%s)
              and not exists (
                  select 1 from unnest(%s::text[], %s::text[]) as keep(video_id, canonical_name)
                  where keep.video_id = v.video_id and keep.canonical_name = v.canonical_name)
            returning v.video_id
            """,
            (registry.video_ids(), video_ids, names),
        )
        return len(rows)

    def counts(self) -> dict[str, int]:
        row = pg_fetch_one(
            """
            select (select count(*) from speakers) as speakers,
                   (select count(*) from speaker_aliases) as aliases,
                   (select count(*) from video_speakers) as video_speakers
            """
        )
        return {k: int(v) for k, v in (row or {}).items()}
