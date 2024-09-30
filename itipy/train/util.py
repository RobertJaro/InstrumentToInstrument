import logging


def skip_invalid(iterable):
    it = iterable.__iter__()
    #
    while True:
        try:
            yield next(it)
        except StopIteration as ex:
            return
        except (AssertionError, ValueError, Exception) as ex:
            logging.error(str(ex))
            continue


def loop(iterable):
    it = iterable.__iter__()
    #
    while True:
        try:
            yield it.next()
        except StopIteration:
            it = iterable.__iter__()
            yield it.next()
        except Exception as ex:
            logging.error(str(ex))
            continue
