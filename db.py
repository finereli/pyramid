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
    
    model = relationship('Model', back_populates='summaries')


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


def init_db(db_path='pyramid.db'):
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    
    session = get_session(db_path)
    
    old_self = session.query(Model).filter_by(name='self').first()
    if old_self:
        old_self.name = 'assistant'
        old_self.description = BASE_MODELS['assistant']
    
    old_agent = session.query(Model).filter_by(name='agent').first()
    if old_agent:
        old_agent.name = 'assistant'
        old_agent.description = BASE_MODELS['assistant']
    
    for name, description in BASE_MODELS.items():
        existing = session.query(Model).filter_by(name=name).first()
        if not existing:
            session.add(Model(name=name, description=description, is_base=True))
        else:
            existing.description = description
    session.commit()
    session.close()
