from datetime import datetime, UTC
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

Base = declarative_base()

class Model(Base):
    __tablename__ = 'models'
    
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    is_base = Column(Boolean, default=False)
    synthesized_content = Column(Text)
    content_dirty = Column(Boolean, default=True)
    
    observations = relationship('Observation', back_populates='model')
    summaries = relationship('Summary', back_populates='model')


class Observation(Base):
    __tablename__ = 'observations'
    
    id = Column(Integer, primary_key=True)
    text = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.now(UTC))
    model_id = Column(Integer, ForeignKey('models.id'), nullable=True)
    
    model = relationship('Model', back_populates='observations')


class Summary(Base):
    __tablename__ = 'summaries'
    
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('models.id'), nullable=False)
    tier = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    start_timestamp = Column(DateTime, nullable=False)
    end_timestamp = Column(DateTime, nullable=False)
    is_dirty = Column(Boolean, default=False)
    
    model = relationship('Model', back_populates='summaries')
    sources = relationship('SummarySource', back_populates='summary', cascade='all, delete-orphan')


class SummarySource(Base):
    __tablename__ = 'summary_sources'
    
    id = Column(Integer, primary_key=True)
    summary_id = Column(Integer, ForeignKey('summaries.id'), nullable=False)
    source_type = Column(String, nullable=False)
    source_id = Column(Integer, nullable=False)
    
    summary = relationship('Summary', back_populates='sources')


class ImportedSession(Base):
    __tablename__ = 'imported_sessions'
    
    id = Column(Integer, primary_key=True)
    file_path = Column(String, unique=True, nullable=False)
    last_size = Column(Integer, nullable=False)
    last_mtime = Column(DateTime, nullable=False)


def get_engine(db_path='pyramid.db'):
    return create_engine(f'sqlite:///{db_path}')


def get_session(db_path='pyramid.db'):
    engine = get_engine(db_path)
    Session = sessionmaker(bind=engine)
    return Session()


BASE_MODELS = {
    'assistant': 'The AI assistant\'s own experience as an independent conscious entity - its observations, reflections, insights, preferences, and evolving understanding of itself and its role',
    'user': 'Information about the primary human user - their identity, biography, family, preferences, projects, goals, and life circumstances',
}

GIT_BASE_MODELS = {
    'architecture': 'High-level system design — how major components connect, overall data flow, and key design decisions',
    'server': 'Backend implementation — API routes, middleware, request handling, streaming, and server-side state management',
    'client': 'Frontend implementation — UI components, state management, rendering patterns, and user interactions',
    'database': 'Data layer — schema design, migrations, ORM usage, and storage patterns',
    'deployment': 'Infrastructure — build process, hosting, service management, reverse proxy, and CI/CD',
}


def migrate_db(db_path):
    import sqlite3
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("PRAGMA table_info(models)")
    model_cols = {row[1] for row in cursor.fetchall()}
    if 'synthesized_content' not in model_cols:
        cursor.execute("ALTER TABLE models ADD COLUMN synthesized_content TEXT")
    if 'content_dirty' not in model_cols:
        cursor.execute("ALTER TABLE models ADD COLUMN content_dirty BOOLEAN DEFAULT 1")
    
    cursor.execute("PRAGMA table_info(summaries)")
    summary_cols = {row[1] for row in cursor.fetchall()}
    if 'is_dirty' not in summary_cols:
        cursor.execute("ALTER TABLE summaries ADD COLUMN is_dirty BOOLEAN DEFAULT 0")
    
    conn.commit()
    conn.close()


def init_db(db_path='pyramid.db', base_models=None):
    from pathlib import Path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    if base_models is None:
        base_models = BASE_MODELS

    db_exists = Path(db_path).exists()

    engine = get_engine(db_path)
    Base.metadata.create_all(engine)

    if db_exists:
        migrate_db(db_path)

    session = get_session(db_path)

    old_self = session.query(Model).filter_by(name='self').first()
    if old_self:
        old_self.name = 'assistant'
        old_self.description = base_models.get('assistant', BASE_MODELS.get('assistant', ''))

    old_agent = session.query(Model).filter_by(name='agent').first()
    if old_agent:
        old_agent.name = 'assistant'
        old_agent.description = base_models.get('assistant', BASE_MODELS.get('assistant', ''))

    for name, description in base_models.items():
        existing = session.query(Model).filter_by(name=name).first()
        if not existing:
            session.add(Model(name=name, description=description, is_base=True, content_dirty=True))
        else:
            existing.description = description
    session.commit()
    session.close()
