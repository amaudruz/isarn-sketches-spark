package org.isarnproject

import org.apache.spark.sql.functions._
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{avg, col}
import org.isarnproject.sketches.spark.tdigest.TDigestAggregator
import org.isarnproject.sketches.spark.tdigest.infra.TDigest

import scala.util.Random._

object Test {

  println("Hello World!")
  // main method
  def main(args: Array[String]): Unit = {
    // create spark session
    // --conf "spark.driver.extraJavaOptions=--add-opens java.base/sun.nio.ch=ALL-UNNAMED" \
    //   --conf "spark.executor.extraJavaOptions=--add-opens java.base/sun.nio.ch=ALL-UNNAMED"
    val spark = SparkSession.builder
      .master("local[2]")
      .appName("SparkTestSuite").getOrCreate()

    // create dataframe
    val data = spark.createDataFrame(Vector.fill(10001){(scala.util.Random.nextInt(10), scala.util.Random.nextGaussian)})
    data.printSchema()

    val udf = TDigestAggregator.udf[Double](compression =10)
    val agg = data.agg(udf(col("_1")).alias("td1"), udf(col("_2")).alias("td2"))
    agg.write.mode("overwrite").parquet("test.parquet")

    agg.show(10)
    spark.udf.register("quantileUDF", TDigest.quantileUDF(0.9))
    agg.selectExpr("quantileUDF(td1)").show(10)



    //data.select(sum("_1")).show(10, false)
  }

}