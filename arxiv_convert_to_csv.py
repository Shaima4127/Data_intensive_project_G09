from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol

class MRConvert_DataToCSV(MRJob):
    OUTPUT_PROTOCOL = RawValueProtocol
    
    # Get a field from a specific linee     
    def parse_val(self, line, field, extra = 3):
    
        index = line.find(field)
        if index != -1:
            val = line[index + len(field) + extra:]           
            return val[:val.find('"')]
        else:   
            return  str(None)

    def mapper(self, _, line):
        # Strip the leading and trailing whitespace from the line
        line = line.strip() 
        line = line[1:-1]

        # Extract the values fields that we want to need it in Spark       
        delimiter = '::'
        id = self.parse_val(line, '"id"', 2)       
        authors = delimiter + self.parse_val(line, '"authors"', 2) + delimiter
        
        title = self.parse_val(line, '"title"', 2)+ delimiter
        abstract = self.parse_val(line, '"abstract"')+ delimiter
        journal_ref = self.parse_val(line, '"journal_ref"')+ delimiter
        categories = self.parse_val(line, '"categories"', 2)+ delimiter
        update_date =  self.parse_val(line, '"update_date"', 2)
        
        
        All_Fields = id + authors +  title + abstract + journal_ref + categories  +update_date
        
        yield _, All_Fields     
    

if __name__ == '__main__':
    MRConvert_DataToCSV.run()
