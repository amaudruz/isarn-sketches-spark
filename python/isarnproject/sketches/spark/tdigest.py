import sys
import random
import itertools as it
from bisect import bisect_left, bisect_right
import struct
from io import BytesIO

from pyspark.sql.types import (
    UserDefinedType,
    StructField,
    StructType,
    ArrayType,
    DoubleType,
    IntegerType,
    BinaryType,
)
from pyspark.sql.column import Column, _to_java_column, _to_seq
from pyspark.context import SparkContext

__all__ = [
    "tdigestQuantileUDF",
    "tdigestIntUDF",
    "tdigestLongUDF",
    "tdigestFloatUDF",
    "tdigestDoubleUDF",
    "tdigestMLVecUDF",
    "tdigestMLLibVecUDF",
    "tdigestIntArrayUDF",
    "tdigestLongArrayUDF",
    "tdigestFloatArrayUDF",
    "tdigestDoubleArrayUDF",
    "tdigestReduceUDF",
    "tdigestArrayReduceUDF",
    "TDigest",
]


def tdigestQuantileUDF(col, quantile: float):
    """
    Return a UDF for extracting the given quantile out of a TDigest column.

    :param quantile: the quantile between 0 and 1
    """
    sc = SparkContext._active_spark_context
    tdapply = sc._jvm.org.isarnproject.sketches.spark.tdigest.infra.TDigest.quantileUDF(
        quantile
    ).apply
    return Column(tdapply(_to_seq(sc, [col], _to_java_column)))


def tdigestIntUDF(col, compression=100.0, maxDiscrete=0):
    """
    Return a UDF for aggregating a column of integer data.

    :param col: name of the column to aggregate
    :param compression: T-Digest compression parameter (default 0.5)
    :param maxDiscrete: maximum unique discrete values to store before reverting to
        continuous (default 0)
    """
    sc = SparkContext._active_spark_context
    tdapply = sc._jvm.org.isarnproject.sketches.spark.tdigest.functions.tdigestIntUDF(
        compression, maxDiscrete
    ).apply
    return Column(tdapply(_to_seq(sc, [col], _to_java_column)))


def tdigestLongUDF(col, compression=100.0, maxDiscrete=0):
    """
    Return a UDF for aggregating a column of long integer data.

    :param col: name of the column to aggregate
    :param compression: T-Digest compression parameter (default 0.5)
    :param maxDiscrete: maximum unique discrete values to store before reverting to
        continuous (default 0)
    """
    sc = SparkContext._active_spark_context
    tdapply = sc._jvm.org.isarnproject.sketches.spark.tdigest.functions.tdigestLongUDF(
        compression, maxDiscrete
    ).apply
    return Column(tdapply(_to_seq(sc, [col], _to_java_column)))


def tdigestFloatUDF(col, compression=100.0, maxDiscrete=0):
    """
    Return a UDF for aggregating a column of (single precision) float data.

    :param col: name of the column to aggregate
    :param compression: T-Digest compression parameter (default 0.5)
    :param maxDiscrete: maximum unique discrete values to store before reverting to
        continuous (default 0)
    """
    sc = SparkContext._active_spark_context
    tdapply = sc._jvm.org.isarnproject.sketches.spark.tdigest.functions.tdigestFloatUDF(
        compression, maxDiscrete
    ).apply
    return Column(tdapply(_to_seq(sc, [col], _to_java_column)))


def tdigestDoubleUDF(col, compression=100.0, maxDiscrete=0):
    """
    Return a UDF for aggregating a column of double float data.

    :param col: name of the column to aggregate
    :param compression: T-Digest compression parameter (default 0.5)
    :param maxDiscrete: maximum unique discrete values to store before reverting to
        continuous (default 0)
    """
    sc = SparkContext._active_spark_context
    tdapply = (
        sc._jvm.org.isarnproject.sketches.spark.tdigest.functions.tdigestDoubleUDF(
            compression, maxDiscrete
        ).apply
    )
    return Column(tdapply(_to_seq(sc, [col], _to_java_column)))


def tdigestMLVecUDF(col, compression=100.0, maxDiscrete=0):
    """
    Return a UDF for aggregating a column of ML Vector data.

    :param col: name of the column to aggregate
    :param compression: T-Digest compression parameter (default 0.5)
    :param maxDiscrete: maximum unique discrete values to store before reverting to
        continuous (default 0)
    """
    sc = SparkContext._active_spark_context
    tdapply = sc._jvm.org.isarnproject.sketches.spark.tdigest.functions.tdigestMLVecUDF(
        compression, maxDiscrete
    ).apply
    return Column(tdapply(_to_seq(sc, [col], _to_java_column)))


def tdigestMLLibVecUDF(col, compression=100.0, maxDiscrete=0):
    """
    Return a UDF for aggregating a column of MLLib Vector data.

    :param col: name of the column to aggregate
    :param compression: T-Digest compression parameter (default 0.5)
    :param maxDiscrete: maximum unique discrete values to store before reverting to
        continuous (default 0)
    """
    sc = SparkContext._active_spark_context
    tdapply = (
        sc._jvm.org.isarnproject.sketches.spark.tdigest.functions.tdigestMLLibVecUDF(
            compression, maxDiscrete
        ).apply
    )
    return Column(tdapply(_to_seq(sc, [col], _to_java_column)))


def tdigestIntArrayUDF(col, compression=100.0, maxDiscrete=0):
    """
    Return a UDF for aggregating a column of integer-array data.

    :param col: name of the column to aggregate
    :param compression: T-Digest compression parameter (default 0.5)
    :param maxDiscrete: maximum unique discrete values to store before reverting to
        continuous (default 0)
    """
    sc = SparkContext._active_spark_context
    tdapply = (
        sc._jvm.org.isarnproject.sketches.spark.tdigest.functions.tdigestIntArrayUDF(
            compression, maxDiscrete
        ).apply
    )
    return Column(tdapply(_to_seq(sc, [col], _to_java_column)))


def tdigestLongArrayUDF(col, compression=100.0, maxDiscrete=0):
    """
    Return a UDF for aggregating a column of long-integer array data.

    :param col: name of the column to aggregate
    :param compression: T-Digest compression parameter (default 0.5)
    :param maxDiscrete: maximum unique discrete values to store before reverting to
        continuous (default 0)
    """
    sc = SparkContext._active_spark_context
    tdapply = (
        sc._jvm.org.isarnproject.sketches.spark.tdigest.functions.tdigestLongArrayUDF(
            compression, maxDiscrete
        ).apply
    )
    return Column(tdapply(_to_seq(sc, [col], _to_java_column)))


def tdigestFloatArrayUDF(col, compression=100.0, maxDiscrete=0):
    """
    Return a UDF for aggregating a column of (single-precision) float array data.

    :param col: name of the column to aggregate
    :param compression: T-Digest compression parameter (default 0.5)
    :param maxDiscrete: maximum unique discrete values to store before reverting to
        continuous (default 0)
    """
    sc = SparkContext._active_spark_context
    tdapply = (
        sc._jvm.org.isarnproject.sketches.spark.tdigest.functions.tdigestFloatArrayUDF(
            compression, maxDiscrete
        ).apply
    )
    return Column(tdapply(_to_seq(sc, [col], _to_java_column)))


def tdigestDoubleArrayUDF(col, compression=100.0, maxDiscrete=0):
    """
    Return a UDF for aggregating a column of double array data.

    :param col: name of the column to aggregate
    :param compression: T-Digest compression parameter (default 0.5)
    :param maxDiscrete: maximum unique discrete values to store before reverting to
        continuous (default 0)
    """
    sc = SparkContext._active_spark_context
    tdapply = (
        sc._jvm.org.isarnproject.sketches.spark.tdigest.functions.tdigestDoubleArrayUDF(
            compression, maxDiscrete
        ).apply
    )
    return Column(tdapply(_to_seq(sc, [col], _to_java_column)))


def tdigestReduceUDF(col, compression=100.0, maxDiscrete=0):
    """
    Return a UDF for aggregating a column of t-digests.

    :param col: name of the column to aggregate
    :param compression: T-Digest compression parameter (default 0.5)
    :param maxDiscrete: maximum unique discrete values to store before reverting to
        continuous (default 0)
    """
    sc = SparkContext._active_spark_context
    tdapply = (
        sc._jvm.org.isarnproject.sketches.spark.tdigest.functions.tdigestReduceUDF(
            compression, maxDiscrete
        ).apply
    )
    return Column(tdapply(_to_seq(sc, [col], _to_java_column)))


def tdigestArrayReduceUDF(col, compression=100.0, maxDiscrete=0):
    """
    Return a UDF for aggregating a column of t-digest vectors.

    :param col: name of the column to aggregate
    :param compression: T-Digest compression parameter (default 0.5)
    :param maxDiscrete: maximum unique discrete values to store before reverting to
        continuous (default 0)
    """
    sc = SparkContext._active_spark_context
    tdapply = (
        sc._jvm.org.isarnproject.sketches.spark.tdigest.functions.tdigestArrayReduceUDF(
            compression, maxDiscrete
        ).apply
    )
    return Column(tdapply(_to_seq(sc, [col], _to_java_column)))


class TDigestUDT(UserDefinedType):
    @classmethod
    def sqlType(cls):
        return BinaryType()

    @classmethod
    def module(cls):
        return "isarnproject.sketches.udt.tdigest"

    @classmethod
    def scalaUDT(cls):
        return "org.apache.spark.isarnproject.sketches.udtdev.TDigestUDT"

    def simpleString(self):
        return "tdigest"

    def serialize(self, obj):
        if isinstance(obj, TDigest):
            return obj.serialize()
        else:
            raise TypeError("cannot serialize %r of type %r" % (obj, type(obj)))

    def deserialize(self, datum):
        return TDigest.deserialize(datum)


class TDigest(object):
    """
    A T-Digest sketch of a cumulative numeric distribution.
    This is a "read-only" python mirror of org.isarnproject.sketches.java.TDigest which supports
    all cdf and sampling methods, but does not currently support update with new data. It is
    assumed to have been produced with a t-digest aggregating UDF, also exposed in this package.
    """

    # Because this is a value and not a function, TDigestUDT has to be defined above,
    # and in the same file.
    __UDT__ = TDigestUDT()

    def __init__(
        self,
        compression,
        maxDiscrete,
        cent,
        mass,
        format_tag,
        min_val=None,
        max_val=None,
        total_weight=None,
    ):
        self.compression = float(compression)
        self.maxDiscrete = int(maxDiscrete)
        assert self.compression > 0.0, "compression must be > 0"
        assert self.maxDiscrete >= 0, "maxDiscrete must be >= 0"
        self._cent = [float(v) for v in cent]
        self._mass = [float(v) for v in mass]
        assert len(self._mass) == len(
            self._cent
        ), "cluster mass and cent must have same dimension"
        self.nclusters = len(self._cent)
        # Current implementation is "read only" so we can just store cumulative sum here.
        # To support updating, 'csum' would need to become a Fenwick tree array
        self._csum = list(it.accumulate(self._mass))

        self.format_tag = format_tag
        self.min_val = min_val or min(cent)
        self.max_val = max_val or max(cent)
        self.total_weight = total_weight or sum(mass)

    def __repr__(self):
        return "TDigest(%s, %s, %s, %s)" % (
            repr(self.compression),
            repr(self.maxDiscrete),
            repr(self._cent),
            repr(self._mass),
        )

    @classmethod
    def deserialize(cls, bytes) -> "TDigest":
        # Wrap the bytes in a BytesIO object for easier reading
        input_buffer = BytesIO(bytes)

        # Read the data in the same order it was written
        format_tag = struct.unpack("B", input_buffer.read(1))[0]

        min_val = struct.unpack("d", input_buffer.read(8))[0]
        max_val = struct.unpack("d", input_buffer.read(8))[0]
        compression = struct.unpack("d", input_buffer.read(8))[0]
        total_weight = struct.unpack("d", input_buffer.read(8))[0]
        centroid_count = struct.unpack("i", input_buffer.read(4))[0]

        # Read the centroids
        centroids = []
        weights = []
        for _ in range(centroid_count):
            mean = struct.unpack("d", input_buffer.read(8))[0]
            centroids.append(mean)

        for i in range(centroid_count):
            weight = struct.unpack("d", input_buffer.read(8))[0]
            weights.append(weight)

        # Check if we have reached the end of the buffer
        if input_buffer.read(1):
            raise ValueError("Extra data found after expected serialized size")

        return TDigest(
            compression=compression,
            maxDiscrete=0,
            cent=centroids,
            mass=weights,
            format_tag=format_tag,
            min_val=min_val,
            max_val=max_val,
            total_weight=total_weight,
        )

    def serialize(self) -> bytes:
        # Calculate the size of the serialized data
        centroid_count = len(self._cent)
        serialized_size_in_bytes = (
            1 + 4 * 8 + 4 + centroid_count * 8 * 2
        )  # 1 byte for format tag, 4 doubles, 1 int, centroids

        # Create a byte buffer to hold the serialized data
        buffer = bytearray(serialized_size_in_bytes)

        # Pack the data into the buffer
        offset = 0
        struct.pack_into("B", buffer, offset, self.format_tag)
        offset += 1
        struct.pack_into("d", buffer, offset, self.min_val)
        offset += 8
        struct.pack_into("d", buffer, offset, self.max_val)
        offset += 8
        struct.pack_into("d", buffer, offset, self.compression)
        offset += 8
        struct.pack_into("d", buffer, offset, self.total_weight)
        offset += 8
        struct.pack_into("i", buffer, offset, centroid_count)
        offset += 4

        # Pack the centroids
        for mean in self._cent:
            struct.pack_into("d", buffer, offset, mean)
            offset += 8

        for weight in self._mass:
            struct.pack_into("d", buffer, offset, weight)
            offset += 8

        return bytes(buffer)

    def mass(self):
        """
        Total mass accumulated by this TDigest
        """
        if len(self._csum) == 0:
            return 0.0
        return self._csum[-1]

    def size(self):
        """
        Number of clusters in this TDigest
        """
        return len(self._cent)

    def isEmpty(self):
        """
        Returns True if this TDigest is empty, False otherwise
        """
        return len(self._cent) == 0

    def __reduce__(self):
        return (
            self.__class__,
            (
                self.compression,
                self.maxDiscrete,
                self._cent,
                self._mass,
            ),
        )

    def _lmcovj(self, m):
        assert self.nclusters >= 2
        assert (m >= 0.0) and (m <= self.mass())
        return bisect_left(self._csum, m)

    def _rmcovj(self, m):
        assert self.nclusters >= 2
        assert (m >= 0.0) and (m <= self.mass())
        return bisect_right(self._csum, m) - 1

    def _rcovj(self, x):
        return bisect_right(self._cent, x) - 1

    # emulates behavior from isarn java TDigest, which computes
    # cumulative sum via a Fenwick tree
    def _ftSum(self, j):
        if j < 0:
            return 0.0
        if j >= self.nclusters:
            return self.mass()
        return self._csum[j]

    def cdf(self, xx):
        """
        Return CDF(x) of a numeric value x, with respect to this TDigest CDF sketch.
        """
        x = float(xx)
        j1 = self._rcovj(x)
        if j1 < 0:
            return 0.0
        if j1 >= self.nclusters - 1:
            return 1.0
        j2 = j1 + 1
        c1 = self._cent[j1]
        c2 = self._cent[j2]
        tm1 = self._mass[j1]
        tm2 = self._mass[j2]
        s = self._ftSum(j1 - 1)
        d1 = 0.0 if (j1 == 0) else tm1 / 2.0
        m1 = s + d1
        m2 = m1 + (tm1 - d1) + (tm2 if (j2 == self.nclusters - 1) else tm2 / 2.0)
        m = m1 + (x - c1) * (m2 - m1) / (c2 - c1)
        return min(m2, max(m1, m)) / self.mass()

    def cdfInverse(self, qq):
        """
        Given a value q on [0,1], return the value x such that CDF(x) = q.
        Returns NaN for any q > 1 or < 0, or if this TDigest is empty.
        """
        q = float(qq)
        if (q < 0.0) or (q > 1.0):
            return float("nan")
        if self.nclusters == 0:
            return float("nan")
        if self.nclusters == 1:
            return self._cent[0]
        if q == 0.0:
            return self._cent[0]
        if q == 1.0:
            return self._cent[self.nclusters - 1]
        m = q * self.mass()
        j1 = self._rmcovj(m)
        j2 = j1 + 1
        c1 = self._cent[j1]
        c2 = self._cent[j2]
        tm1 = self._mass[j1]
        tm2 = self._mass[j2]
        s = self._ftSum(j1 - 1)
        d1 = 0.0 if (j1 == 0) else tm1 / 2.0
        m1 = s + d1
        m2 = m1 + (tm1 - d1) + (tm2 if (j2 == self.nclusters - 1) else tm2 / 2.0)
        x = c1 + (m - m1) * (c2 - c1) / (m2 - m1)
        return min(c2, max(c1, x))

    def cdfDiscrete(self, xx):
        """
        return CDF(x) for a numeric value x, assuming the sketch is representing a
        discrete distribution.
        """
        x = float(xx)
        j = self._rcovj(x)
        return self._ftSum(j) / self.mass()

    def cdfDiscreteInverse(self, qq):
        """
        Given a value q on [0,1], return the value x such that CDF(x) = q, assuming
        the sketch is represenging a discrete distribution.
        Returns NaN for any q > 1 or < 0, or if this TDigest is empty.
        """
        q = float(qq)
        if (q < 0.0) or (q > 1.0):
            return float("nan")
        if self.nclusters == 0:
            return float("nan")
        if self.nclusters == 1:
            return self._cent[0]
        m = q * self.mass()
        j = self._lmcovj(m)
        return self._cent[j]

    def samplePDF(self):
        """
        Return a random sampling from the sketched distribution, using inverse
        transform sampling, assuming a continuous distribution.
        """
        return self.cdfInverse(random.random())

    def samplePMF(self):
        """
        Return a random sampling from the sketched distribution, using inverse
        transform sampling, assuming a discrete distribution.
        """
        return self.cdfDiscreteInverse(random.random())

    def sample(self):
        """
        Return a random sampling from the sketched distribution, using inverse
        transform sampling, assuming a discrete distribution if the number of
        TDigest clusters is <= maxDiscrete, and a continuous distribution otherwise.
        """
        if self.maxDiscrete <= self.nclusters:
            return self.cdfDiscreteInverse(random.random())
        return self.cdfInverse(random.random())
