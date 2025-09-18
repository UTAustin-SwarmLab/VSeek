from data.lvb import LongVideoBench

if __name__ == "__main__":
    dataset = LongVideoBench()
    dataset.save_it_as_vseek_data(desired_interval_in_sec=1)
