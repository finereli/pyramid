import pytest
from datetime import datetime, UTC, timedelta
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db import Base, Model, Observation, Summary
from summarize import get_day_start, chunk_observations, get_observations_by_day


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


def test_get_day_start():
    dt = datetime(2025, 6, 15, 14, 30, 45, tzinfo=UTC)
    day_start = get_day_start(dt)
    assert day_start.hour == 0
    assert day_start.minute == 0
    assert day_start.second == 0
    assert day_start.day == 15


def test_get_observations_by_day(session):
    day1 = datetime(2025, 6, 15, 10, 0, 0)
    day1_later = datetime(2025, 6, 15, 18, 0, 0)
    day2 = datetime(2025, 6, 16, 12, 0, 0)
    
    session.add(Observation(text='Obs 1', timestamp=day1, importance=5))
    session.add(Observation(text='Obs 2', timestamp=day1_later, importance=5))
    session.add(Observation(text='Obs 3', timestamp=day2, importance=5))
    session.commit()
    
    by_day = get_observations_by_day(session)
    assert len(by_day) == 2
    
    days = sorted(by_day.keys())
    assert len(by_day[days[0]]) == 2
    assert len(by_day[days[1]]) == 1


def test_chunk_observations_small(session):
    obs = Observation(text='Short observation', timestamp=datetime.now(UTC), importance=5)
    session.add(obs)
    session.commit()
    
    chunks = chunk_observations([obs])
    assert len(chunks) == 1
    assert len(chunks[0]) == 1
