import pytest

from tennis.core.queue import SqliteQueue


@pytest.fixture
def queue(tmp_path):
    return SqliteQueue(f"sqlite:///{tmp_path}/q.db")


def test_receive_hides_the_message_until_its_visibility_timeout(queue):
    queue.enqueue("job-1", "PROBE_VIDEO")

    msg = queue.receive(visibility_timeout_s=60)

    assert (msg.job_id, msg.stage, msg.attempts) == ("job-1", "PROBE_VIDEO", 1)
    assert queue.receive(visibility_timeout_s=60) is None


def test_an_unacked_message_is_redelivered_after_its_lease_expires(queue):
    queue.enqueue("job-1", "BALL_TRACKING")
    first = queue.receive(visibility_timeout_s=0)  # worker "crashes" without acking

    again = queue.receive(visibility_timeout_s=60)

    assert again.id == first.id
    assert again.attempts == 2


def test_acked_messages_are_never_redelivered(queue):
    queue.enqueue("job-1", "PROBE_VIDEO")
    msg = queue.receive(visibility_timeout_s=0)

    queue.ack(msg)

    assert queue.receive(visibility_timeout_s=60) is None


def test_nack_makes_the_message_visible_again_immediately(queue):
    queue.enqueue("job-1", "PROBE_VIDEO")
    msg = queue.receive(visibility_timeout_s=600)

    queue.nack(msg, "boom")

    again = queue.receive(visibility_timeout_s=600)
    assert again.id == msg.id


def test_messages_are_received_in_fifo_order(queue):
    queue.enqueue("job-1", "A")
    queue.enqueue("job-1", "B")

    assert queue.receive(60).stage == "A"
    assert queue.receive(60).stage == "B"


def test_two_queue_instances_on_one_database_share_messages(tmp_path):
    url = f"sqlite:///{tmp_path}/shared.db"
    producer, consumer = SqliteQueue(url), SqliteQueue(url)  # UI process vs worker process

    producer.enqueue("job-1", "PROBE_VIDEO")

    assert consumer.receive(60).stage == "PROBE_VIDEO"
