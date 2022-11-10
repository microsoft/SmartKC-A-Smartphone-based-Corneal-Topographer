package com.example.kt.injection

import android.content.Context
import androidx.room.Room
import com.example.kt.data.manager.KTDatabase
import dagger.Module
import dagger.Provides
import dagger.hilt.InstallIn
import dagger.hilt.android.qualifiers.ApplicationContext
import dagger.hilt.components.SingletonComponent
import javax.inject.Singleton

@Module
@InstallIn(SingletonComponent::class)
class DatabaseModule {
    @Provides
    @Singleton
    fun providesKTDatabase(@ApplicationContext context: Context): KTDatabase {
        return Room.databaseBuilder(context, KTDatabase::class.java, "kt.db").build()
    }
}