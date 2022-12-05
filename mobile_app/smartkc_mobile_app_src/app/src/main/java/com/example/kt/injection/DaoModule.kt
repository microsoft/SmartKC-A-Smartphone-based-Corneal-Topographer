package com.example.kt.injection

import com.example.kt.data.local.daos.FileDao
import com.example.kt.data.manager.KTDatabase
import dagger.Module
import dagger.Provides
import dagger.Reusable
import dagger.hilt.InstallIn
import dagger.hilt.components.SingletonComponent

@Module
@InstallIn(SingletonComponent::class)
class DaoModule {
    @Provides
    @Reusable
    fun provideFileDao(ktDatabase: KTDatabase): FileDao {
        return ktDatabase.fileDao()
    }
}