import time
from contextlib import contextmanager
from prepros.dataprep import *
from traintest.testtrain import *


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    print(" ")


def main():
    with timer("Feature Engineering"):
        print("Feature Engineering started")
        df = feature_eng()

    with timer("Data Preprocessing"):
        print("Data Preproccessing Started")
        df = dataprep(df)

    with timer("TrainTest"):
        print("TrainTest Started")
        finalresult = tratest(df)
        print(finalresult)


main()