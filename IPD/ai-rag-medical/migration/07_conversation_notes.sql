create table if not exists conversation_notes (
  id bigserial primary key,
  stase_slug text not null,
  session_id text not null,
  query text not null,
  note_title text not null,
  note_summary text,
  note_markdown text not null default '',
  disease_name text,
  library_catalog_id bigint,
  note_status text not null default 'saved' check (
    note_status in (
      'draft',
      'saved',
      'ready_to_promote',
      'promoted_to_library',
      'archived',
      'deleted'
    )
  ),
  draft_answer jsonb not null default '{}'::jsonb,
  evidence_summary jsonb not null default '[]'::jsonb,
  citation_quality jsonb,
  retrieval_metadata jsonb,
  match_candidates jsonb,
  citation_precision numeric(6,4),
  promoted boolean not null default false,
  promoted_at timestamptz,
  promoted_library_article_id bigint,
  deleted_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists idx_conversation_notes_session_id
  on conversation_notes (session_id, created_at desc);

create index if not exists idx_conversation_notes_stase_status
  on conversation_notes (stase_slug, note_status, created_at desc);

create index if not exists idx_conversation_notes_library_catalog_id
  on conversation_notes (library_catalog_id);

create index if not exists idx_conversation_notes_updated_at
  on conversation_notes (updated_at desc);

create or replace function touch_conversation_notes_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

drop trigger if exists trg_conversation_notes_updated_at on conversation_notes;
create trigger trg_conversation_notes_updated_at
before update on conversation_notes
for each row
execute function touch_conversation_notes_updated_at();
