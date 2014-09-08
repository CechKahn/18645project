package mapred.hashtagsim;

import java.io.IOException;

import mapred.util.Tokenizer;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.IntWritable;

/*
 * HashtagMapper -- First time mapper to extract hash tag and
 *					feature from dataset
 */
public class HashtagMapper extends Mapper<LongWritable, Text, Text, MapWritable> {
	//hash map to use as a combiner
	private MapWritable occurMap = new MapWritable();

	@Override
	protected void map(LongWritable key, Text value, Context context)
			throws IOException, InterruptedException {
		String line = value.toString();
		String[] words = Tokenizer.tokenize(line);

		occurMap.clear();

		for(String word : words)
			if(word.startsWith("#")) {
				Text textWord = new Text(word);
				if(occurMap.containsKey(word)) {
					//save count in hash
					IntWritable count = (IntWritable)occurMap.get(textWord);
					count.set(count.get() + 1);
				}
				else {
					occurMap.put(textWord, new IntWritable(1));
				}
			}

		for(String word : words) 
			if(word.startsWith("#") == false) {
				context.write(new Text(word), occurMap);
			}

	}
}
