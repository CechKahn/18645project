package mapred.hashtagsim;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.io.MapWritable;
import org.apache.hadoop.io.ObjectWritable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.IntWritable;


/*
 * HashtagReduer -- First time reduer to reduce data from hashmapper
 					the output is feature #hashtag1:count1;#hahstag2:count2
 */
public class HashtagReducer extends Reducer<Text, MapWritable, Text, Text> {

	@Override
	protected void reduce(Text key, Iterable<MapWritable> value, Context context)
			throws IOException, InterruptedException {
		//use hash map to store data and count
		Map<String, Integer> counts = new HashMap<String, Integer>();

		for (MapWritable occurMap : value) {
			Set<Writable> keys = occurMap.keySet();
			for(Writable k : keys) {
				String w = ((Text)k).toString();
				if(counts.containsKey(w)) {
					int count = counts.get(w);
					count++;
					counts.put(w, count);
				}
				else {
					counts.put(w, 1);
				}
			}
		}

		//output result in format: feature #hashtag1:count1;#hahstag2:count2...
		StringBuilder builder = new StringBuilder();
		for (Map.Entry<String, Integer> e : counts.entrySet())
			builder.append(e.getKey() + ":" + e.getValue() + ";");

		context.write(key, new Text(builder.toString()));
	}
}
