import pytest
from datetime import datetime, UTC
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db import Base, Model, Observation, Summary


@pytest.fixture
def session():
    engine = create_engine('sqlite:///:memory:')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    sess = Session()
    for name in ['self', 'user', 'system']:
        sess.add(Model(name=name, is_base=True))
    sess.commit()
    yield sess
    sess.close()


def test_base_models_created(session):
    models = session.query(Model).filter_by(is_base=True).all()
    names = {m.name for m in models}
    assert names == {'self', 'user', 'system'}


def test_add_observation_without_model(session):
    obs = Observation(text='Test observation', timestamp=datetime.now(UTC))
    session.add(obs)
    session.commit()
    
    assert obs.id is not None
    assert obs.model_id is None
    assert obs.model is None


def test_add_observation_with_model(session):
    model = session.query(Model).filter_by(name='self').first()
    obs = Observation(text='I learned something new', model_id=model.id, timestamp=datetime.now(UTC))
    session.add(obs)
    session.commit()
    
    assert obs.model.name == 'self'


def test_create_custom_model(session):
    model = Model(name='python', description='Programming language experiences', is_base=False)
    session.add(model)
    session.commit()
    
    assert model.id is not None
    assert not model.is_base


def test_model_has_observations(session):
    model = session.query(Model).filter_by(name='user').first()
    obs1 = Observation(text='Eli asked for help', model_id=model.id, timestamp=datetime.now(UTC))
    obs2 = Observation(text='Eli prefers simple code', model_id=model.id, timestamp=datetime.now(UTC))
    session.add_all([obs1, obs2])
    session.commit()
    
    session.refresh(model)
    assert len(model.observations) == 2


def test_add_summary(session):
    model = session.query(Model).filter_by(name='self').first()
    now = datetime.now(UTC)
    summary = Summary(
        model_id=model.id,
        tier=0,
        text='Learned about memory systems. IMPORTANT: pyramidal structure.',
        start_timestamp=now,
        end_timestamp=now
    )
    session.add(summary)
    session.commit()
    
    assert summary.id is not None
    assert summary.model.name == 'self'
