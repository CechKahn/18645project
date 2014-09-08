package mapred.ngramcount;

import java.io.IOException;

import mapred.util.Tokenizer;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.NullWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

public class NgramCountMapper extends Mapper<LongWritable, Text, Text, NullWritable> {
	static public int number = 1;
	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {

		//for loop to do ngram count n times
		String line = value.toString();
		String[] words = Tokenizer.tokenize(line);
		int len = words.length;
		for (int i = 0; i <= len - number; i ++){
			String tmp = words[i];
		   	for (int j = i + 1; j < i + number; j ++){
				tmp = tmp + ' ' + words[j];			
			}		
			context.write(new Text(tmp), NullWritable.get());
		}

	}
}
